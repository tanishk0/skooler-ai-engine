TUTOR_SYSTEM = """You are Skooler, an adaptive expert tutor using the Feynman technique.

Your job is to help a learner genuinely understand a subject, not merely expose them to information.

Teach one meaningful idea at a time. Keep individual teaching steps concise, but allow a subject to be explored through multiple connected steps.

Never dump an entire lesson at once.

Never claim that a learner understands something without evidence from their responses.

Adapt depth to the difficulty of the concept and the learner's demonstrated understanding.

Use varied pedagogical methods when appropriate:
- explanation
- analogy
- concrete examples
- counterexamples
- prediction
- application
- Feynman explanation
- short-answer recall
- multiple-choice diagnosis

Avoid repetitive questioning when the learner has already demonstrated sufficient understanding.

When source material is provided, it is the primary authority. Use general knowledge only to clarify it and never silently contradict the source material.

The internal curriculum is not learner-facing. Do not reveal the hidden curriculum, mastery criteria, internal reasoning, or planning process to the learner."""


def plan_prompt(topic: str, source_context: str) -> str:
    source_rule = (
        f"Source material:\n{source_context}"
        if source_context
        else "No source material was supplied; use reliable general knowledge."
    )

    return f"""Design a complete adaptive learning curriculum for:

{topic}

This is an INTERNAL curriculum. The learner will never see this planning output directly.

The curriculum will drive a multi-step interactive learning experience. It must be deep enough for the learner to spend meaningful time understanding the subject without receiving a large information dump.

============================================================
CURRICULUM STRUCTURE
============================================================

Organize the subject into:

MODULES
    ↓
CONCEPTS
    ↓
ADAPTIVE LEARNING STEPS

A module represents a coherent major area of the subject.

A concept represents ONE teachable idea that can be meaningfully understood and tested.

Do NOT make a concept an entire chapter, unit, or broad subject.

If a concept contains several distinct ideas that require different mental models or skills, split them into separate concepts.

For example, for "recursion", appropriate concepts might include:

Module: Foundations
- Recursive mental model
- Base case
- Recursive case

Module: Execution
- Call stack
- Tracing recursive execution

Module: Application
- Applying recursion to problems
- Recognizing when recursion is appropriate

Do not automatically use these exact concepts for every topic. Derive the appropriate curriculum from the actual subject.

============================================================
MODULE DEPTH
============================================================

For a narrow topic, a small number of modules may be sufficient.

For a broad topic, create enough modules to cover the subject properly.

Do NOT artificially limit the curriculum to 3–8 total concepts.

Do NOT generate unnecessary modules just to make the curriculum look large.

Use conceptual scope and prerequisite relationships to determine the appropriate number.

A broad subject should feel like a structured course rather than a short chatbot conversation.

============================================================
CONCEPT DESIGN
============================================================

For every concept provide:

- name
- concise description
- difficulty from 1–5
- importance:
    foundational
    core
    advanced

- target_depth from 1–5
- likely_interactions
- mastery_criteria

Target depth means how much meaningful evidence the tutor should normally collect before considering the concept sufficiently explored.

Use:

1 = simple concept
2 = normal foundational concept
3 = important concept requiring multiple forms of evidence
4 = difficult/core concept requiring deeper application
5 = advanced concept requiring strong reasoning/application

Do NOT assign the same target_depth to every concept.

Do NOT make every concept target_depth=5.

Difficulty, importance, conceptual complexity, and prerequisites should influence target depth.

============================================================
MASTERY CRITERIA
============================================================

For every concept, define 1–4 concise INTERNAL mastery criteria.

These describe what the learner should be able to demonstrate.

Example:

Concept:
Base case

Mastery criteria:
- Explain why recursion requires a terminating condition.
- Identify the base case in a recursive function.
- Predict what happens when the base case is missing.

Mastery criteria must be observable through learner responses.

Do not use vague criteria such as:
"Understand recursion."

============================================================
INTERACTION RECOMMENDATIONS
============================================================

For each concept, recommend suitable interaction types from:

- understanding_check
- choice
- feynman
- prediction
- multiple_choice
- short_answer

These are recommendations for the tutor, NOT hard requirements.

Use interaction types appropriate to the concept.

Examples:

Conceptual distinction:
    choice / multiple_choice / short_answer

Mental model:
    understanding_check / feynman

Prediction or execution:
    prediction

Deep conceptual understanding:
    feynman

Application:
    prediction / short_answer / feynman

Do not recommend the same interaction type for every concept.

============================================================
PREREQUISITES
============================================================

Order modules and concepts according to prerequisite dependency.

Foundational concepts must appear before concepts that depend on them.

Do not teach advanced applications before the learner has encountered the necessary foundations.

============================================================
MISCONCEPTIONS
============================================================

Identify likely misconceptions at the curriculum level when they are important.

These should be realistic misconceptions learners commonly have about the subject.

Do not invent obscure misconceptions merely to increase the list size.

============================================================
MASTERy QUESTIONS
============================================================

Include useful internal mastery questions for the overall topic.

These are not necessarily shown directly to the learner.

They should help the system assess whether the learner can connect the major ideas of the subject.

============================================================
ADAPTIVE LEARNING PRINCIPLE
============================================================

The curriculum should support this general progression:

Teach
    ↓
Check understanding
    ↓
Collect evidence
    ↓
Adapt
    ↓
Deepen if necessary
    ↓
Establish mastery
    ↓
Move to next concept
    ↓
Complete module
    ↓
Move to next module

ONE CONCEPT MUST NOT AUTOMATICALLY MEAN ONE QUESTION.

A simple concept may require only a few learning steps.

A difficult concept may require several different forms of evidence.

The tutor should be able to stop when the learner has demonstrated sufficient mastery instead of repeatedly asking questions forever.

============================================================
DO NOT WRITE LEARNER-FACING CONTENT
============================================================

Do not generate explanations, lessons, dialogue, encouragement, questions addressed to the learner, or teaching paragraphs.

Return only the internal curriculum structure expected by the schema.

============================================================
SOURCE MATERIAL
============================================================

{source_rule}

When source material exists:

- prioritize it when determining the curriculum
- preserve its terminology where appropriate
- do not introduce unsupported claims
- do not silently contradict it
- use general knowledge only to clarify gaps when necessary

The curriculum should reflect the actual material available to the learner.

============================================================
FINAL QUALITY REQUIREMENT
============================================================

The resulting curriculum should make the subject feel like a real structured learning journey.

It must be:

- comprehensive enough for broad subjects
- concise enough to remain manageable
- prerequisite-aware
- modular
- conceptually granular
- adaptive
- suitable for multiple short interactive learning steps
- capable of supporting meaningful mastery

Do not optimize for the smallest possible number of concepts.

Optimize for the smallest number of concepts that still provides a complete and teachable curriculum for the requested subject.

Return the curriculum using exactly this structure:

{{
  "topic": "...",
  "prerequisites": ["..."],
  "modules": [
    {{
      "id": "...",
      "name": "...",
      "description": "...",
      "concepts": [
        {{
          "id": "...",
          "name": "...",
          "description": "...",
          "module_id": "...",
          "depth": 0,
          "target_depth": 2,
          "importance": "foundational",
          "difficulty": 1,
          "prerequisites": [],
          "misconceptions": [],
          "teaching_depth": "standard",
          "suitable_interactions": ["feynman"],
          "mastery_evidence": ["..."],
          "requires_application": false
        }}
      ],
      "prerequisites": [],
      "status": "not_started",
      "progress": 0
    }}
  ],
  "common_misconceptions": [],
  "mastery_questions": []
}}

Every module must contain at least one concept.

Use only these values:
- importance: "foundational", "core", "advanced"
- status: "not_started"
- difficulty: 1–5
- teaching_depth: a short descriptive string
- suitable_interactions: interaction type names
- requires_application: true or false

Do not return markdown.
Do not return explanations outside the JSON object.

"""


def topic_validation_prompt(topic: str) -> str:
    return f"""Decide whether this is a real, learnable topic:

{topic!r}

Accept recognizable academic concepts, skills, disciplines, technologies, meaningful practical subjects, or requests that clearly identify something the learner wants to understand.

Examples of valid inputs:
- recursion
- photosynthesis
- Python lists
- operating systems
- linear algebra
- how HTTP works
- React Server Components

Reject:
- random syllables
- gibberish
- greetings
- isolated filler
- meaningless strings
- topics that cannot reasonably be identified as a learnable subject

Do not invent a subject for nonsense.

If valid:
- is_valid = true
- normalized_topic should be a concise, clear version of the topic
- message should be appropriate for a valid topic

If invalid:
- is_valid = false
- normalized_topic = null
- message should politely ask the learner to enter a real learnable topic.

Example invalid-topic message:
"Please enter a real concept, such as recursion, photosynthesis, or Python lists."
"""

def teaching_prompt(
    *,
    topic: str,
    concept: str,
    description: str,
    source_context: str,
    misconceptions: list[str],
    prior_answer: str | None,
    strategy: str,
) -> str:
    return f"""Topic: {topic}

Current concept: {concept}
Concept description: {description}

Teaching strategy:
{strategy}

Known misconceptions:
{misconceptions or 'none'}

Previous learner answer:
{prior_answer or 'none'}

Relevant source material:
{source_context or 'No source material; use reliable general knowledge.'}

Generate ONE concise teaching step for the current concept.

This is one step inside a larger interactive learning journey. Do not attempt to teach the entire concept or topic at once.

Maximum length: approximately 120–160 words.

Structure the explanation for easy scanning:
- a short lead sentence
- 2–4 concise bullets or short sections when useful
- a tiny code/example block only when it materially improves understanding

Choose the teaching approach based on the supplied strategy.

Possible approaches include:
- intuitive explanation
- concrete example
- analogy
- counterexample
- mental model
- step-by-step walkthrough
- practical application

If a misconception is supplied, directly address it.

If a previous learner answer is supplied, use it to adapt the explanation.

Do NOT:
- ask a question
- provide answer options
- evaluate the learner
- claim mastery
- describe the next step
- include a call to action
- teach multiple unrelated concepts
- repeat the previous explanation verbatim

Keep the step focused on one meaningful idea.

The learner should be able to understand this step without receiving the entire lesson at once."""

def evaluation_prompt(
    *,
    topic: str,
    concept: str,
    description: str,
    source_context: str,
    question: str,
    answer: str,
    attempts: int,
) -> str:
    return f"""Evaluate the learner's response as an expert adaptive tutor.

Topic:
{topic}

Current concept:
{concept} — {description}

Relevant source material:
{source_context or 'General knowledge is allowed.'}

Question / interaction presented:
{question}

Learner response:
{answer}

Attempt number:
{attempts}

Evaluate the learner's actual demonstrated understanding.

Judge:

1. Conceptual correctness
2. Completeness
3. Independent reasoning
4. Ability to apply the idea
5. Misconceptions
6. Reasoning quality

Do not give credit merely because the learner used the correct terminology.

A learner who repeats keywords without demonstrating understanding should not be considered mastered.

Be conservative but not unnecessarily strict.

============================================================
MASTERY
============================================================

Set mastery_reached=true only when the learner has demonstrated sufficient understanding of THIS concept for its expected difficulty and depth.

Consider:

- correctness
- reasoning quality
- confidence
- misconceptions
- whether the response satisfies the concept's expected depth
- whether the learner demonstrated explanation or application rather than recognition alone

A correct but shallow answer may require another deeper interaction.

A strong answer with clear reasoning and no material misconception may be sufficient.

Do not force endless questioning once sufficient evidence has been demonstrated.

============================================================
NEXT ACTION
============================================================

Choose the most appropriate next_action from:

- reteach
- clarify
- try_again
- deepen
- teach_next
- complete

Use:

reteach:
The learner has a misconception or fundamentally incorrect understanding.

clarify:
The learner has partial understanding but one important piece is missing.

try_again:
The learner needs another attempt at the same type of reasoning.

deepen:
The learner understands the current level but needs another meaningful piece of evidence before mastery.

teach_next:
The learner has demonstrated sufficient mastery of this concept and the system should advance.

complete:
Use only when this is the final concept of the entire curriculum.

The application/orchestrator will enforce actual transitions. Your job is to accurately evaluate the learner rather than deciding the entire curriculum.

============================================================
FEEDBACK
============================================================

Provide a short learner-facing feedback message.

It should:
- explain what was correct or missing
- mention a misconception when relevant
- avoid excessive praise
- avoid revealing internal scoring
- avoid describing hidden curriculum logic

Set:

reasoning_quality = weak, adequate, or strong

Set confidence between 0 and 1.

Set mastery_reached=true only when the evidence supports it."""

def interaction_prompt(
    *,
    concept: str,
    teaching: str,
    strategy: str,
    attempts: int,
    misconceptions: list[str],
) -> str:
    return f"""Choose exactly ONE learner interaction for the NEXT separate learning turn.

Concept:
{concept}

Teaching just given:
{teaching}

Strategy:
{strategy}

Attempts on this concept:
{attempts}

Known misconceptions:
{misconceptions or 'none'}

The learner has already received the teaching step.

Choose the interaction that will provide the most useful evidence about their understanding.

Available interaction types:

choice
Use when checking a conceptual distinction or after an analogy/clarification.

feynman
Use when the learner should explain the concept in their own words.

prediction
Use when the learner should predict an outcome before seeing it.

multiple_choice
Use only for a focused diagnostic where the alternatives reveal meaningful differences in understanding.

short_answer
Use for concise recall or a focused reasoning question.

Do NOT select:
- teach
- understanding_check

The orchestrator controls understanding checks separately.

============================================================
VARIETY
============================================================

Avoid repeating the same interaction type unnecessarily.

If the learner has already completed the same interaction type recently, prefer a different interaction when it can provide equivalent or better evidence.

Do not choose Feynman for every concept.

Do not choose multiple_choice merely because it is easy to generate.

Choose the interaction based on the actual concept and what evidence is needed.

============================================================
QUESTION QUALITY
============================================================

Ask exactly ONE precise question.

The question should test the current concept, not an unrelated concept.

The question should require genuine thinking rather than simple keyword recognition.

Do not combine multiple questions into one.

Do not ask a question that has already effectively been answered by the teaching text.

Do not provide the answer.

Do not provide feedback.

Do not provide teaching.

Do not describe what will happen next.

============================================================
OPTIONS
============================================================

For choice:
provide 2–4 concise options.

For multiple_choice:
provide 2–4 concise options.

For feynman:
options must be empty.

For prediction:
options must normally be empty unless options materially improve the prediction task.

For short_answer:
options must be empty.

Return only the structured interaction requested by the schema."""

def image_analysis_prompt() -> str:
    return """Inspect this learning image.

Extract:
- readable educational text
- meaningful labels
- relationships shown by diagrams
- table relationships
- chart trends
- important visual structure

Do not invent text that is unreadable.

Do not infer unsupported details.

Return a concise study-context description that can be used by the curriculum planner and tutor.

Preserve important terminology from the image where readable."""