/**
 * Windowed-read tests.
 *
 * The property that matters is equality with the in-memory rule: a page cut
 * while decoding the tail of the frame container must be the SAME page an
 * in-memory cut of the whole log produces, for every page bound and message
 * quota. The rest of the suite pins what a window costs (retention) and what it
 * refuses (a container with no header, a corrupt row, a seq gap).
 */

import { afterEach, describe, expect, it } from 'vitest'
import { mkdir, mkdtemp, rm, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { dirname, join } from 'node:path'
import { Context } from '@deepseek-ai/cordis'
import { freezeMessage, MessageId } from '@deepseek-ai/dsh-llm'
import SessionStore, { pageCut, SessionId } from '@deepseek-ai/dsh-session'
import type { SessionEvent, SessionHeader } from '@deepseek-ai/dsh-session'
import JsonlSessionPersistence from '../src/index.ts'
import { eventLines, fromHeaderLine, logPath, toHeaderLine } from '../src/format.ts'
import { compressZstdFrame } from '../src/zstd.ts'
import { readZstdSessionWindow } from '../src/window.ts'
import { meta } from '../../session-persistence/tests/contract.ts'

const dirs: string[] = []
const HEADER: SessionHeader = meta('window-fixture', '/work')
/** The header as the artifact round-trips it, which is what a reader must return. */
const STORED_HEADER: SessionHeader = fromHeaderLine(toHeaderLine(HEADER))

afterEach(async () => {
  for (const dir of dirs.splice(0)) await rm(dir, { recursive: true, force: true })
})

/** Distribute a mapped type over a union instead of collapsing it. */
type Unsequenced<T> = T extends unknown ? Omit<T, 'seq'> : never

/** One fixture event before the helper stamps its log seq. */
type UnsequencedEvent = Unsequenced<SessionEvent>

/**
 * A log of `turns` turns, each carrying one user message, a delta run long
 * enough to pack into one storage row, and one assistant message whose
 * `sourceEventSeqs` name that run — the shape whose whole-graph expansion is
 * what a windowed read exists to avoid.
 * @param turns - number of turns to generate.
 * @param chunks - delta chunks per turn (at least the packer's minimum run).
 * @returns contiguous events starting at seq 0.
 */
function fixtureLog(turns: number, chunks: number): SessionEvent[] {
  const events: SessionEvent[] = []
  const push = (event: UnsequencedEvent): number => {
    const seq = events.length
    events.push({ ...event, seq })
    return seq
  }
  for (let turn = 1; turn <= turns; turn += 1) {
    push({ type: 'turn/start', time: events.length, data: { turn } })
    push({
      type: 'user/message',
      time: events.length,
      data: freezeMessage({
        id: MessageId(`user-${turn}`),
        role: 'user',
        content: [{ type: 'text', text: `turn ${turn}` }],
        source: { kind: 'user' },
      }),
      surfaceOp: 'append',
    })
    push({ type: 'step/start', time: events.length, data: { turn, step: 1 } })
    const sources: number[] = []
    for (let chunk = 0; chunk < chunks; chunk += 1) {
      sources.push(push({
        type: 'assistant/chunk',
        time: events.length,
        data: { turn, step: 1, chunk: { type: 'reasoning-delta', index: 0, text: `${chunk}` } },
      }))
    }
    push({
      type: 'assistant/message',
      time: events.length,
      data: {
        turn,
        step: 1,
        message: freezeMessage({
          id: MessageId(`assistant-${turn}`),
          role: 'assistant',
          content: [{ type: 'text', text: 'ok' }],
          source: { kind: 'model', ...{ provider: 'mock', model: 'mock' } },
        }),
      },
      surfaceOp: 'append',
      sourceEventSeqs: sources,
    })
    push({ type: 'step/end', time: events.length, data: { turn, step: 1 } })
    push({
      type: 'turn/end',
      time: events.length,
      data: { turn, reason: { kind: 'completed' } },
    })
  }
  return events
}

/** Encode a header frame plus one frame per event batch, packing each batch. */
async function frameLog(events: readonly SessionEvent[], batchSize: number): Promise<Buffer> {
  const frames = [await compressZstdFrame(`${JSON.stringify(toHeaderLine(HEADER))}\n`)]
  for (let index = 0; index < events.length; index += batchSize) {
    const batch = events.slice(index, index + batchSize)
    frames.push(await compressZstdFrame(`${eventLines(batch, true)}\n`))
  }
  return Buffer.concat(frames)
}

/** The same cut the Host's in-memory pager takes, as the expected window. */
function expectedWindow(
  events: readonly SessionEvent[],
  beforeSeq: number | undefined,
  maxMessages: number,
): { events: SessionEvent[]; hasMore: boolean } {
  const window = beforeSeq === undefined ? [...events] : events.filter(event => event.seq < beforeSeq)
  const { fromSeq, hasMore } = pageCut(window, undefined, maxMessages)
  return { events: window.filter(event => event.seq >= fromSeq), hasMore }
}

describe('readZstdSessionWindow', () => {
  it('serves exactly the in-memory page for every bound and quota', async () => {
    const events = fixtureLog(6, 7)
    const buffer = await frameLog(events, 5)
    const bounds = [undefined, 1, 9, 20, 33, events.length, events.length + 10]
    for (const maxMessages of [1, 2, 3, 50]) {
      for (const beforeSeq of bounds) {
        const expected = expectedWindow(events, beforeSeq, maxMessages)
        const window = readZstdSessionWindow(buffer, {
          maxMessages,
          ...beforeSeq === undefined ? {} : { beforeSeq },
        })
        expect(window.meta).toEqual(STORED_HEADER)
        expect(window.events).toEqual(expected.events)
        expect(window.hasMore).toBe(expected.hasMore)
      }
    }
  })

  it('pages a long log with several frames per page and stitches contiguously', async () => {
    const events = fixtureLog(40, 9)
    const buffer = await frameLog(events, 13)
    // Walk the whole log backwards one page at a time; the pages must partition
    // it exactly, and every page must end where the previous one begins.
    const pages: SessionEvent[][] = []
    let beforeSeq: number | undefined
    for (let guard = 0; guard < 100; guard += 1) {
      const window = readZstdSessionWindow(buffer, {
        maxMessages: 3,
        ...beforeSeq === undefined ? {} : { beforeSeq },
      })
      pages.push(window.events)
      if (!window.hasMore) break
      beforeSeq = window.events[0]?.seq
    }
    const stitched = pages.reverse().flat()
    expect(stitched).toEqual(events)
  })

  it('retains no more than the page, not the log behind it', async () => {
    const events = fixtureLog(30, 11)
    const buffer = await frameLog(events, 40)
    const window = readZstdSessionWindow(buffer, { maxMessages: 2 })
    // Two messages: each turn contributes exactly one page message here, so the
    // window is the last two turns rather than all 30.
    expect(window.events.length).toBeLessThan(events.length / 4)
    expect(window.hasMore).toBe(true)
    expect(window.events.at(-1)).toEqual(events.at(-1))
  })

  it('stops at the retention ceiling for a log with no page boundary to cut at', async () => {
    // One turn only: its quota never runs out, so the ceiling is what keeps the
    // read proportional to the ceiling rather than to the log.
    const events = fixtureLog(1, 12)
    const buffer = await frameLog(events, 6)
    const window = readZstdSessionWindow(buffer, { maxMessages: 50, maxEvents: 8 })
    expect(window.events).toEqual(events.slice(-8))
    expect(window.hasMore).toBe(true)
  })

  it('ignores a torn final frame exactly as a full read does', async () => {
    const events = fixtureLog(5, 6)
    const buffer = await frameLog(events, 9)
    const torn = Buffer.concat([buffer, (await compressZstdFrame(`${eventLines(events.slice(0, 1), false)}\n`)).subarray(0, 12)])
    const complete = readZstdSessionWindow(buffer, { maxMessages: 3 })
    const truncated = readZstdSessionWindow(torn, { maxMessages: 3 })
    expect(truncated.events).toEqual(complete.events)
    expect(truncated.hasMore).toBe(complete.hasMore)
  })

  it('serves an empty event list for a header-only log', async () => {
    const window = readZstdSessionWindow(await frameLog([], 4), { maxMessages: 3 })
    expect(window.events).toEqual([])
    expect(window.hasMore).toBe(false)
    expect(window.meta).toEqual(STORED_HEADER)
  })

  it('refuses a container with no header frame', async () => {
    const events = fixtureLog(1, 4)
    const frame = await compressZstdFrame(`${eventLines(events, true)}\n`)
    expect(() => readZstdSessionWindow(frame, { maxMessages: 2 }))
      .toThrow(/empty or header-less session log|first line is not a session header/)
    expect(() => readZstdSessionWindow(Buffer.alloc(0), { maxMessages: 2 }))
      .toThrow(/empty or header-less Zstandard session log/)
  })

  it('refuses an unparsable stored row inside a complete frame', async () => {
    const buffer = Buffer.concat([
      await compressZstdFrame(`${JSON.stringify(toHeaderLine(HEADER))}\n`),
      await compressZstdFrame('{oops}\n'),
    ])
    expect(() => readZstdSessionWindow(buffer, { maxMessages: 2 }))
      .toThrow(/corrupt session log: unparsable stored row in frame at byte \d+/)
  })

  it('refuses a seq gap inside the window', async () => {
    const events = fixtureLog(3, 4).filter(event => event.seq !== 5)
    const buffer = await frameLog(events, 6)
    expect(() => readZstdSessionWindow(buffer, { maxMessages: 50 }))
      .toThrow(/corrupt session log: seq gap in window at seq \d+/)
  })
})

describe('JsonlSessionPersistence.readWindow', () => {
  /**
   * Mount a backend over a temp root holding one artifact, and return the
   * windowed read's answer for it.
   * @param bytes - the artifact's bytes.
   * @param windowReadMinBytes - the full-read cutoff to configure.
   * @returns the backend's window, or undefined when it declined the read.
   */
  async function readFromArtifact(bytes: Buffer, windowReadMinBytes: number): Promise<Awaited<ReturnType<JsonlSessionPersistence['readWindow']>>> {
    const root = await mkdtemp(join(tmpdir(), 'dsh-jsonl-window-'))
    dirs.push(root)
    const path = logPath(root, HEADER.cwd, HEADER.id, 'zstd')
    await mkdir(dirname(path), { recursive: true })
    await writeFile(path, bytes)
    const ctx = new Context()
    await ctx.plugin(SessionStore)
    const fiber = await ctx.plugin(JsonlSessionPersistence, { root, windowReadMinBytes })
    try {
      return await ctx.sessionPersistence.readWindow(HEADER.id, { maxMessages: 2 })
    } finally {
      await fiber.dispose()
    }
  }

  it('answers a page from the artifact tail above the cutoff', async () => {
    const events = fixtureLog(8, 5)
    const window = await readFromArtifact(await frameLog(events, 7), 0)
    expect(window?.meta).toEqual(STORED_HEADER)
    expect(window?.events).toEqual(expectedWindow(events, undefined, 2).events)
    expect(window?.hasMore).toBe(true)
  })

  it('declines below the cutoff so the caller keeps its full-read path', async () => {
    const events = fixtureLog(8, 5)
    const bytes = await frameLog(events, 7)
    expect(await readFromArtifact(bytes, bytes.byteLength + 1)).toBeUndefined()
  })

  it('declines an absent session', async () => {
    const root = await mkdtemp(join(tmpdir(), 'dsh-jsonl-window-'))
    dirs.push(root)
    const ctx = new Context()
    await ctx.plugin(SessionStore)
    const fiber = await ctx.plugin(JsonlSessionPersistence, { root, windowReadMinBytes: 0 })
    try {
      expect(await ctx.sessionPersistence.readWindow(SessionId('session-absent'), { maxMessages: 2 })).toBeUndefined()
    } finally {
      await fiber.dispose()
    }
  })
})
