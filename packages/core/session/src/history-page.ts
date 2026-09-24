/**
 * Message-boundary pagination over one transcript's append-origin surface.
 *
 * The rule lives here, apart from any reader, because two readers must agree
 * on it: the in-memory pager that cuts an already-materialized event list, and
 * the streamed window reader that decides its cut while decoding a log it
 * never holds whole. A page's cut is the oldest seq of the
 * `maxMessages`-th newest append-origin message, so a page boundary can never
 * land inside a message assembled from packed deltas.
 * @module @deepseek-ai/dsh-session/history-page
 */

import { isAppendSurfaceEvent } from './surface.ts'
import type { SessionEvent } from './types.ts'

/** Conversation message event types: the unit `maxMessages` counts. */
const PAGE_MESSAGE_TYPES: ReadonlySet<string> = new Set(['user/message', 'assistant/message'])

/**
 * Whether one event counts as one page message: an append-origin conversation
 * message. Replacement copies restate a shadowed range for the model alone and
 * never entered the transcript a reader sees, so they consume no quota.
 * @param event - candidate log event.
 * @returns true when the event is one conversation message of a page.
 */
export function isPageMessage(event: SessionEvent): boolean {
  return PAGE_MESSAGE_TYPES.has(event.type) && isAppendSurfaceEvent(event)
}

/**
 * The first seq belonging to one page message's group. An Assistant message
 * assembled from delta chunks carries {@link SessionEvent.sourceEventSeqs}
 * naming the chunk run it was folded from; the group starts at its earliest
 * source, which is what keeps a cut from splitting the message in two.
 * @param event - a page message, as identified by {@link isPageMessage}.
 * @returns the inclusive seq a page containing this message must start at.
 */
export function pageGroupStart(event: SessionEvent): number {
  const sources = (event as { sourceEventSeqs?: number[] }).sourceEventSeqs
  let start = event.seq
  if (sources !== undefined) {
    for (const source of sources) {
      if (source < start) start = source
    }
  }
  return start
}

/**
 * Count `maxMessages` append-origin messages backwards from the end of the
 * window and report where the resulting page starts.
 *
 * A window shorter than `maxMessages` messages starts its page at seq 0 — the
 * whole window is one page, and no older event exists to fetch. A `beforeSeq`
 * bound is exclusive: events at or above it belong to the page a reader
 * already holds and neither consume quota nor enter this page.
 * @param events - the log, or the pre-bounded window, oldest first.
 * @param beforeSeq - exclusive upper bound in event seq; omitted pages the tail.
 * @param maxMessages - maximum append-origin messages one page may contain.
 * @returns the page's inclusive first seq and whether older events remain.
 */
export function pageCut(
  events: readonly SessionEvent[],
  beforeSeq: number | undefined,
  maxMessages: number,
): { fromSeq: number; hasMore: boolean } {
  let count = 0
  let fromSeq = 0
  for (let index = events.length - 1; index >= 0; index -= 1) {
    // Non-empty by the loop bound; the assertion documents that for the checker.
    // oxlint-disable-next-line typescript/no-non-null-assertion
    const event = events[index]!
    if (beforeSeq !== undefined && event.seq >= beforeSeq) continue
    if (!isPageMessage(event)) continue
    count += 1
    if (count >= maxMessages) {
      fromSeq = pageGroupStart(event)
      break
    }
  }
  return { fromSeq, hasMore: fromSeq > 0 }
}
