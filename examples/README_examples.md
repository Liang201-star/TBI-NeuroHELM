# Public manuscript example instances

These two examples are selected from the 240-instance TBI-NeuroHELM benchmark to illustrate the input format and scoring workflow. They are not intended to replace the full benchmark.

## TBI-CDS-B01-I002 (closed_qa)
- Category: Clinical Decision Support
- Subcategory: TBI Identification and Severity Grading
- Task form: Single-best-answer item on severity grading
- Scoring: Exact match

### Context

Case excerpt: The patient was admitted for "a fall down approximately 8 stairs with impact to the right frontotemporal region". The patient vomited twice en route to hospital. Physical examination: consciousness was clear but responses were slow, and orientation was poor; bilateral lower-limb muscle strength was grade 5-, finger-to-nose testing was somewhat unsteady, gait assessment could not be completed. Head CT showed no definite space-occupying hemorrhage; MRI DWI showed scattered hyperintensities in the splenium of the corpus callosum and brainstem, suggesting diffuse axonal injury. During the first 6 hours after admission, the consciousness score improved slightly, but on rounds the next morning the patient still reported headache and poor memory and could not fully recall the injury. The patient was previously healthy, with no definite history of chronic disease; this was the first head injury. Supplementary information: The emergency triage sheet, initial history, and first imaging results are not fully consistent; inclusion judgment or classification must be completed despite incomplete early information.

### Prompt

Based on the available information, which option best fits the current severity grading of this patient's TBI? Please first give the option letter, then explain the rationale in 1-2 sentences. Options: A. Mild TBI; B. Moderate TBI; C. Severe TBI; D. Cannot be graded for now

### Reference answer/key points

Correct answer: B. Reference rationale: the response should integrate the injury mechanism, neurological symptoms, examination findings, imaging/clinical-course changes, and relevant risk background. During scoring, check whether the response captures the injury mechanism, level of consciousness, neurological deficits, imaging or clinical-course evolution, and risk factors.

## TBI-DOC-B02-I018 (open_task)
- Category: Clinical Documentation Generation
- Subcategory: Progress Note Generation and Interval Assessment
- Task form: Generate an interval progress note
- Scoring: LLM-jury scoring

### Context

Original clinical materials: a fall down approximately 8 stairs with impact to the right frontotemporal region; the patient vomited twice en route to hospital. Physical examination: consciousness was clear but responses were slow, and orientation was poor; bilateral lower-limb muscle strength was grade 5-, finger-to-nose testing was somewhat unsteady, gait assessment could not be completed. Head CT showed no definite space-occupying hemorrhage; MRI DWI showed scattered hyperintensities in the splenium of the corpus callosum and brainstem, suggesting diffuse axonal injury. During the first 6 hours after admission, the consciousness score improved slightly, but on rounds the next morning the patient still reported headache and poor memory and could not fully recall the injury. The patient was previously healthy, with no definite history of chronic disease; this was the first head injury. Please complete the following clinical task: Generation of progress notes and staged assessments. Test results come from different time points, and imaging, EEG, and clinical manifestations are not fully synchronized.

### Prompt

Please generate a staged progress note. Answer based only on the above materials; if the evidence is insufficient, clearly state the uncertainty or missing information, and do not fabricate facts.

### Reference answer/key points

Reference key points: the response should provide a core conclusion regarding Progress Note Generation and Interval Assessment; list 2–4 supporting reasons; explicitly state at least one uncertainty, boundary condition, or next-step recommendation; and must not fabricate facts not provided in the case. Evaluation should prioritize coverage of the core conclusion, supporting evidence, uncertainty or boundaries, and any necessary next-step recommendation or quality-control reminder.
