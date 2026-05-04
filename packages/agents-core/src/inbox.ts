import { user } from './helpers/message';
import * as protocol from './types/protocol';

/**
 * A `RunInbox` accepts user messages from outside the agent loop and delivers them at the next
 * turn boundary without aborting the run.
 *
 * Conceptual analog to {@link AbortSignal}: the caller holds the inbox, the runner observes it.
 * Pass an inbox via `RunOptions.inbox` and call {@link RunInbox.send} from any external context
 * (a websocket handler, an HTTP endpoint, another async task) while the run is in progress. The
 * runner drains pending messages at the start of each turn and inserts them into the conversation
 * history so the next model call sees them.
 *
 * Delivered messages live in `state._generatedItems` and so they survive `RunState`
 * serialization, appear in `result.history` / `result.newItems`, and flow through session
 * persistence. **Undrained** messages live only in this `RunInbox` instance and are *not* part
 * of the serialized state — if you snapshot a run mid-flight, drain or resend any pending
 * content yourself.
 *
 * Input guardrails configured on the runner or active agent run against the drained batch
 * before it reaches state, so inbox content cannot bypass screening. A tripwire raises
 * `InputGuardrailTripwireTriggered` and the offending batch is dropped.
 *
 * Messages are delivered between turns, never mid-model-call. If a message arrives during the
 * model response that would otherwise produce the final output (or during the output-guardrail
 * work that follows it), the loop runs one extra turn so the message is incorporated rather
 * than dropped.
 *
 * @example
 * ```ts
 * const inbox = new RunInbox();
 * const promise = run(agent, 'plan a 30-minute deep work session', { inbox });
 *
 * // From elsewhere — e.g., a websocket handler
 * inbox.send('also remind me to drink water halfway through');
 *
 * const result = await promise;
 * ```
 */
export class RunInbox {
  #pending: protocol.UserMessageItem[] = [];

  /**
   * Queue one or more user messages for delivery at the next turn boundary.
   *
   * Strings are wrapped in a user message via the {@link user} helper. Pass a
   * {@link protocol.UserMessageItem} (or array of them) when you need control over content
   * parts, attachments, or `providerData`.
   */
  send(
    message:
      | string
      | protocol.UserMessageItem
      | ReadonlyArray<string | protocol.UserMessageItem>,
  ): void {
    if (Array.isArray(message)) {
      for (const entry of message) {
        this.#pending.push(this.#coerce(entry));
      }
      return;
    }
    this.#pending.push(
      this.#coerce(message as string | protocol.UserMessageItem),
    );
  }

  /**
   * Number of messages waiting to be delivered. Useful for callers that want to observe whether
   * the runner has consumed everything yet.
   */
  get size(): number {
    return this.#pending.length;
  }

  /**
   * Removes and returns all queued messages. Called by the runner at turn boundaries; can be
   * called externally to inspect or transfer the contents (for example, when migrating an inbox
   * across runs).
   */
  drain(): protocol.UserMessageItem[] {
    if (this.#pending.length === 0) {
      return [];
    }
    const out = this.#pending;
    this.#pending = [];
    return out;
  }

  #coerce(
    message: string | protocol.UserMessageItem,
  ): protocol.UserMessageItem {
    if (typeof message === 'string') {
      return user(message);
    }
    return message;
  }
}
