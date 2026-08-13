# Security FAQ

## Where is customer data stored?

In the EU (Frankfurt) for EU customers and in the US (Oregon) for everyone else.
Region is fixed at account creation and cannot be changed afterwards without opening a
migration ticket, because moving it means rewriting every foreign key that references
the account.

## Is data encrypted?

Yes, in transit and at rest. Transport is TLS 1.3. At rest, volumes use AES-256 with
keys held in a managed KMS. Note this is encryption at rest against physical media
access — our systems can still read your records in order to serve them, which is a
different property from end-to-end encryption and worth being clear about.

## Who can see my records internally?

Access is role-scoped and every read is logged. Support engineers can see order
metadata but not payment instrument details, which are held by our payment processor
and referenced by token. Nobody at the company has access to full card numbers.

## How long do you keep data after account closure?

Operational records are deleted 90 days after closure. Financial records are retained
for seven years, because tax law requires it — deletion requests cannot override a
statutory retention period, and we will say so rather than quietly ignoring the
request.

## Do you use customer data for model training?

No. Support transcripts and order records are not used to train any model. If that
ever changes it would be an opt-in with notice, not a policy update buried in terms.

## What happens during a breach?

Affected customers are notified within 72 hours of confirmation, which matches the
GDPR requirement. The notice states what was accessed, when, and what we have done —
before the investigation is complete, if necessary. Waiting for a tidy narrative
delays the only thing that actually helps, which is customers rotating credentials.

## Can I request a penetration test?

Enterprise accounts can, with 30 days notice and a signed scope agreement. Testing
against shared infrastructure is not permitted, so tests run against a dedicated
staging environment provisioned for the engagement.
