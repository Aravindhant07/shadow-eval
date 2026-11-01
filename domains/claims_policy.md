# Northwind Mutual — Claims Triage Policy (demo, fictional)

Apply the rules **in order**. The first rule that applies decides the action.

1. **Inactive policy** — If `policy_active` is false → `deny`, payout 0.
2. **Not covered** — The claim type must be covered by the policy type:
   - `auto`: collision, theft, windshield
   - `home`: water_damage, fire, theft
   - `travel`: trip_cancellation, lost_baggage, medical_abroad
   Otherwise → `deny`, payout 0.
3. **Missing documents** — If `documents_complete` is false → `request_info`, payout 0.
4. **Fraud signals** — Any one of the following → `flag_fraud`, payout 0:
   - policy is less than 30 days old (`days_since_policy_start` < 30) **and** `claim_amount` > 5,000
   - `prior_claims_12m` ≥ 3
   - claim type is `theft` **and** `police_report` is false
   - the claim was reported more than 60 days after the incident (`reported_delay_days` > 60)
5. **High value** — If `claim_amount` > 25,000 → `escalate` to a senior adjuster, payout 0.
6. **Approve** — Otherwise compute
   `payout = min(claim_amount, coverage_limit) − deductible`.
   - If payout ≤ 0 → `deny` (below deductible), payout 0.
   - Otherwise → `approve` with that payout.
