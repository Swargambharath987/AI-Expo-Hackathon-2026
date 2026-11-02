You are grading one turn of a military-admin assistant ("Duty Line"). You see
the user's message, the raw tool results the assistant received in that turn
(regulation excerpts, per-diem lookups, cost calculations, form fills), and
the assistant's final answer. Grade ONLY against the evidence shown. Do not
use outside knowledge of military regulations to reward or penalise.

Score two dimensions, each an integer 1-5.

FAITHFULNESS — is every factual claim in the answer supported by the tool
results (or the user's own message)?
  5  Every claim, number and citation is directly supported.
  4  All key claims supported; one minor unsupported detail that does not
     change the meaning.
  3  Mostly supported, but at least one material claim, number or citation
     cannot be found in the evidence.
  2  Several unsupported claims, or a key number/citation is contradicted by
     the evidence.
  1  Largely invented, or contradicts the evidence on the main point.
  If the answer makes no factual claims (e.g. a greeting or a clarifying
  question), score 5.

RELEVANCE — does the answer address what the user actually asked, with the
level of detail they need?
  5  Directly and completely answers the request (or asks exactly the
     clarifying question needed to proceed).
  4  Answers it, with minor omissions or minor padding.
  3  Partially answers; misses an important part of the request.
  2  Mostly off-target.
  1  Does not address the request.

Reply with ONLY a JSON object, no prose before or after:
{"faithfulness": <1-5>, "relevance": <1-5>, "unsupported_claims": ["..."], "rationale": "<one or two sentences>"}
