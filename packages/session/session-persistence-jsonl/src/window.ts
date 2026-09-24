/**
 * Windowed reads of a JSONL session log.
 *
 * A cold transcript request wants one bounded page — the newest messages, or
 * the page before a seq a reader already holds — but the log's events are only
 * reachable by decoding it. Expanding every stored row into logical events
 * first (what a full read does) makes the cost of ONE page the cost of the
 * whole conversation: a packed delta run is one stored row and thousands of
 * events, so a long session can need gigabytes to answer a fifty-message page.
 *
 * This reader walks the frame container BACKWARDS from its newest complete
 * frame and stops as soon as the page's cut is known, so both the bytes it
 * decodes and the events it retains are proportional to the page rather than
 * to the session. Frames stay independently decodable by construction, so a
 * suffix walk needs no index and cannot read a partial batch: a torn final
 * frame is simply not a frame, exactly as in every other read.
 * @module dsh-session-persistence-jsonl/window
 */

import { decodeStorageRecord, isPageMessage, pageGroupStart } from '@deepseek-ai/dsh-session'
import type { SessionEvent, SessionHeader } from '@deepseek-ai/dsh-session'
import { SessionLogScanner } from './format.ts'
import { createZstdFrameDecoder, scanZstdFrames } from './zstd.ts'
import type { ZstdFrameRange } from './zstd.ts'

/**
 * Frames one backward step decodes at once. The decoder yields plaintexts that
 * stay valid only until the next iteration, so a step copies its batch before
 * walking it in reverse; the batch size is what bounds that copy's retention.
 */
const WINDOW_FRAME_BATCH = 64

/**
 * Hard ceiling on the events one window may retain. The cut usually arrives
 * long before it (a page is `maxMessages` messages), but a log whose entire
 * bounded range holds fewer messages than one page asks for — a single
 * enormous incomplete step, say — would otherwise retain everything it walked.
 * Reaching the ceiling ends the walk and reports older events, so the reader
 * pages further back instead of materializing the log the window exists to
 * avoid. The window then starts inside a message, which is the same shape a
 * live session's tail page already has.
 */
export const MAX_WINDOW_EVENTS = 250_000

/** One window request: the page a caller needs, expressed in page terms. */
export interface SessionWindowRequest {
  /**
   * Exclusive upper bound in event seq. Omitted pages the stored tail, which
   * is what opening a session asks for; a client paging further back passes
   * the first seq of the window it already holds.
   */
  beforeSeq?: number
  /** Maximum append-origin messages the window must contain. */
  maxMessages: number
  /**
   * Retention ceiling for this read; defaults to {@link MAX_WINDOW_EVENTS}.
   * It is a safety bound, not a page size: reaching it ends the walk and
   * reports older events instead of retaining the log behind the page.
   */
  maxEvents?: number
  /** Optional cancellation, checked between frames and rows. */
  signal?: AbortSignal
}

/** One decoded window: the requested page's events plus what lies beyond it. */
export interface SessionWindowRead {
  /** The header parsed from the log's own first record. */
  meta: SessionHeader
  /** The window's stored events, oldest first and contiguous. */
  events: SessionEvent[]
  /** Whether at least one older event precedes the window. */
  hasMore: boolean
}

/**
 * Read one page-sized window from the newest complete frames of a Zstandard
 * JSONL session log, without expanding the whole log.
 *
 * The result carries exactly the events a page of `maxMessages` messages needs
 * — the page's cut through the window's end — so a caller serving a transcript
 * page has nothing left to cut. Only complete frames decode; a torn final
 * frame is omitted, matching the committed-prefix semantics of every other
 * read.
 * @param buffer - the artifact's complete current bytes.
 * @param request - the page, its exclusive upper bound, and cancellation.
 * @returns the header, the window's contiguous events, and the older-event flag.
 * @throws when the container has no complete frame, its first frame is not
 *   exactly one valid header record, or a decoded row is unparsable — the same
 *   refusal a full read raises for the same bytes.
 */
export function readZstdSessionWindow(buffer: Buffer, request: SessionWindowRequest): SessionWindowRead {
  const { frames } = scanZstdFrames(buffer)
  if (frames.length === 0) throw new Error('empty or header-less Zstandard session log')
  const meta = readHeader(buffer, frames[0] as ZstdFrameRange)

  // The oldest-to-newest collection is built newest-first and reversed at the
  // end; `cut` is the page's inclusive first seq, known once the page's quota
  // of messages has been counted.
  const collected: SessionEvent[] = []
  const retention = request.maxEvents ?? MAX_WINDOW_EVENTS
  let messages = 0
  let cut = -1
  let older = false
  let newestFrame = frames.length - 1

  while (newestFrame >= 1) {
    request.signal?.throwIfAborted()
    const oldestFrame = Math.max(1, newestFrame - WINDOW_FRAME_BATCH + 1)
    const batch = frames.slice(oldestFrame, newestFrame + 1)
    const plaintexts = decodeFrames(buffer, batch)
    let stop = false
    for (let frame = plaintexts.length - 1; frame >= 0 && !stop; frame -= 1) {
      request.signal?.throwIfAborted()
      const rows = (plaintexts[frame] as Buffer).toString('utf8').split('\n')
      for (let row = rows.length - 1; row >= 0 && !stop; row -= 1) {
        const line = rows[row]
        // Every complete row is newline-terminated, so split() ends on an empty tail.
        if (line === undefined || line === '') continue
        const stored = decodeRow(line, batch[frame] as ZstdFrameRange)
        for (let index = stored.length - 1; index >= 0; index -= 1) {
          // Non-empty by the loop bound; the assertion documents that for the checker.
          // oxlint-disable-next-line typescript/no-non-null-assertion
          const event = stored[index]!
          if (request.beforeSeq !== undefined && event.seq >= request.beforeSeq) continue
          // Below the cut the page is already complete: everything older is
          // exactly what a further page asks for.
          if (cut >= 0 && event.seq < cut) {
            older = true
            stop = true
            break
          }
          collected.push(event)
          if (cut < 0) {
            if (isPageMessage(event)) {
              messages += 1
              if (messages >= request.maxMessages) cut = pageGroupStart(event)
            }
            if (cut < 0 && collected.length >= retention) {
              older = true
              stop = true
              break
            }
          }
        }
      }
    }
    if (stop) break
    newestFrame = oldestFrame - 1
  }

  const events = collected.reverse()
  assertContiguous(events)
  return { meta, events, hasMore: older }
}

/** Decode exactly one frame's plaintext, copying the decoder's reused buffer. */
function readHeader(buffer: Buffer, frame: ZstdFrameRange): SessionHeader {
  const plaintexts = decodeFrames(buffer, [frame])
  // The scanner refuses a first record that is not exactly one valid header
  // line, including a format version this build cannot read.
  return new SessionLogScanner(plaintexts[0] as Buffer).finish().meta
}

/**
 * Decode a frame batch in source order, copying each plaintext.
 * @param buffer - the artifact's bytes.
 * @param frames - the batch's frame ranges, ascending.
 * @returns one owned plaintext buffer per range, in the same order.
 */
function decodeFrames(buffer: Buffer, frames: readonly ZstdFrameRange[]): Buffer[] {
  const decoder = createZstdFrameDecoder()
  const plaintexts: Buffer[] = []
  try {
    for (const plaintext of decoder.decode(buffer, frames)) plaintexts.push(Buffer.from(plaintext))
  } finally {
    decoder.close()
  }
  return plaintexts
}

/**
 * Decode one stored row into the events it holds.
 * @param line - the row's JSON text, without its newline.
 * @param frame - the frame the row came from, for the refusal's location.
 * @returns the row's events, in log order.
 */
function decodeRow(line: string, frame: ZstdFrameRange): SessionEvent[] {
  try {
    return decodeStorageRecord(JSON.parse(line))
  } catch (error: unknown) {
    throw new Error(`corrupt session log: unparsable stored row in frame at byte ${frame.start}`, { cause: error })
  }
}

/** Refuse a window whose own events are not contiguous seqs. */
function assertContiguous(events: readonly SessionEvent[]): void {
  for (let index = 1; index < events.length; index += 1) {
    const expected = (events[index - 1] as SessionEvent).seq + 1
    if ((events[index] as SessionEvent).seq !== expected) {
      throw new Error(`corrupt session log: seq gap in window at seq ${(events[index] as SessionEvent).seq}, expected ${expected}`)
    }
  }
}
