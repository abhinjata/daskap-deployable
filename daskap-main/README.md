# Daskap

A community helper bot, powered by Groq AI, LangChain & LangGraph Framework, with Retrieval Augmented Scoring techniques designed to help a select club/community's Organising Committee understand and serve the people of the society better. The model that was used is "llama-3.3-70b-versatile", and a vector database powered by Chroma. This is an agentic AI framework curated to generate content which is emotionally and ideologically relevant to the the person's answers and thought process.

## Why Daskap?

Communities don’t collapse because people stop caring — they collapse because the ties between them become alienated. When members’ voices blur into noise, surveys pile up, and feedback is reduced to numbers on a spreadsheet, the social relation between people appears as a relation between isolated datapoints. The real pulse of the community is lost. That alienation weakens the very fabric that should hold us together.

**Daskap** aims to bridge that gap. It transforms fragmented voices into clear signals of collective need, helping organisers act with empathy and clarity. Think of it as a bridge: between the individual and the collective, between fragmented expression and shared understanding, between alienation and belonging. It uses AI not to replace human connection, but to **restore it**.

---

## Theory

* **Retrieval-Augmented Scoring (RAS):** Ingest messy, scattered feedback and score it for relevance, novelty, and emotional intensity.
* **Emotionally Intelligent Modeling:** Detect underlying emotions and synthesize responses that acknowledge and respect them.
* **Persona & Insight Generation:** Build coherent portraits of community sentiment — making the abstract concrete.
* **Agentic Flow:** Use LangGraph to chain retrieval, scoring, and generation into decision-ready insights for organisers.

---

## Details

* **Models:** llama-3.3-70b-versatile, but am planning to train a Qwen Model (7B, 14B, optionally 32B/72B) for emotional intelligence and nuanced reasoning.
* **Retriever:** Chroma (vector DB) for feedback storage and retrieval.
* **Frameworks:** LangChain + LangGraph for orchestration.
* **Agentic logic:** Multi-step pipeline: Retrieve → Score → Collate → Generate → Rerank → Deliver.

---

## Repository layout

```
Daskap/                # root
├─ DaskapBot/           # main python package (agentic logic), the chain framework was a prototype the graph framework is the latest one 
├─ README.md            # this file
├─ requirements.txt     # dependencies (to be refined)
└─ user_feedback.csv    # sample feedback data (if provided)
```

---

## Execute It Yourself

1. **Clone the repo**

```bash
git clone https://github.com/abhinjata/daskap.git
cd daskap
```

2. **Set up environment**

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

3. **Configure the API**
   Yes, this is not hosted. This is annoying but hey its a work in progress so I'll deploy it later.

   Create a `.env` file with the following:

```
QWEN_MODEL=qwen-14b-instruct
CHROMA_DIR=./chroma_db
GROQ_API_KEY=your_groq_key   
```

4. **Run the agent**

```bash
python -m DaskapBot.main
```

---

## How it works

1. **Retrieve:** Collect feedback snippets from vector DB.
2. **Score:** Use a Qwen-7B scorer to assign relevance, novelty, and emotional intensity scores.
3. **Collate:** Build a context of top-ranked snippets.
4. **Generate:** Pass to Qwen-14B/32B to create empathetic summaries, personas, and action suggestions.
5. **Rerank & Filter:** Apply safety checks and ensure outputs are non-manipulative, respectful, and transparent.
6. **Deliver:** Outputs are returned as structured JSON with emotions, summaries, actions, and suggested empathetic messages.

---

## Example output 

Generated Persona:
 **Persona Name:** Ethan Thompson
**Estimated Age:** 20-22 years old

**Goals:**
Ethan is a passionate and driven individual who is deeply interested in Computer Science. His primary goal is to contribute to the club's true purpose and ideology, which he believes is being slightly deviated from due to the focus on dopamine-oriented events. He aspires to be part of the Organizing Committee (OC) to drive the club's vision forward and promote more technical and project-based activities.

**Behaviour:**
Ethan is an active and engaged member of the community, frequently attending events and participating in discussions. He is not afraid to share his opinions and suggestions for improvement, demonstrating his commitment to the club's growth and success. His high rating of the community (9/10) indicates that he is largely satisfied with the club, but has some areas of concern that he hopes to address.

**Interests:**
Ethan's interests are deeply rooted in Computer Science, and he is enthusiastic about exploring and working on projects related to the field. He values the technical and intellectual aspects of CS and believes that the club should focus more on these areas rather than entertainment-oriented events.

**Noted Suggestions for Improvement:**
Ethan's suggestion to shift the focus from dopamine-oriented events to more nerdy and cool CS projects is a key area for improvement. The club could benefit from incorporating more technical workshops, hackathons, or project-based activities to cater to members like Ethan who are eager to learn and contribute to the field. Additionally, providing more opportunities for members to take on leadership roles or contribute to the OC could help retain talented and motivated individuals like Ethan.

Extracted Suggestions:
 Here are the constructive suggestions the user made for improving the club:
 * Focus more on "nerdy and cool CS projects" rather than dopamine-oriented events like video games.
* The user also expressed a desire to be part of the Organizing Committee (OC) to help push the true ideology of the club ahead, implying that they want to contribute to the club's direction and decision-making process.

---

## Contributing

1. Fork the repo.
2. Create a feature branch.
3. Open a PR with a clear description.

Suggested contributions:

* Add datasets for emotional fine-tuning.
* Extend RAS scoring heuristics.
* Improve LangGraph orchestration.

---

## License

TBD (MIT or similar).

---

**Daskap** — giving communities the tools to truly hear themselves. 

(If this README.md looks ever so slighty corporate or product oriented, it's because I'm using this for an entrepreneurship course's idea. I'd hate to commodifiy this, it's just a cool tool for people to get a voice and for my introverted a** to hear more people without selling all my social energy.)
