import json
import os
from typing import Dict, List

import anthropic
import pandas as pd


class ClaudeJobEvaluator:
    """Evaluates job postings against a resume using Claude API"""

    def __init__(self, model: str = "claude-haiku-4-5"):
        self.client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        self.model = model

    def evaluate(self, resume: str, jobs_df: pd.DataFrame) -> pd.DataFrame:
        results = []

        # System Prompt: Focused on the 4 key metrics
        system_text = """You are a tallented Talent Evaluator. trying to deliver a good fit and or a growth opratunity
        1. Extract the candidate's "Current Baseline" from their RESUME.
        2. Compare strictly against the JOB DESCRIPTION.

        """

        system = [
            {"type": "text", "text": system_text},
            {
                "type": "text",
                "text": f"CANDIDATE RESUME:\n\n{resume}",
                "cache_control": {"type": "ephemeral"},
            },
        ]

        for idx, job in jobs_df.iterrows():
            print(
                f"Evaluating job {idx + 1}/{len(jobs_df)}: {job.get('company', 'Unknown')}"
            )
            result = self._score_job(job, system)
            results.append(result)

        return pd.DataFrame(results)

    def _score_job(self, job: pd.Series, system: List[Dict]) -> Dict:
        prompt = f"""
        JOB DESCRIPTION:
        Company: {job.get("company", "Unknown")}
        Title: {job.get("title", "Unknown")}

        {job.get("description", "")[:6000]}

        INSTRUCTIONS:
        Analyze the job and return a valid JSON object.

        FORMATTING RULES:
        1. "verdict": Keep it short (e.g. "Step Up", "Lateral", "Title Regression").
        2. "tech_stack": Identify the top 5 matches and top 3 gaps.
        3. "tech_stack.verdict": Be enthusiastic if they match well (e.g. "Elite Match", "Perfect Fit").
        4. "one_line_summary": Highlight the trade-off (e.g. "Technical slam dunk, but a step backward in authority.")

        JSON STRUCTURE:
        {{
          "match_scores": {{
              "skills_match": <int 1-10>,
              "career_level_alignment": <int 1-10>,
              "experience_relevance": <int 1-10 (Domain/Problem fit)>,
              "culture_fit": <int 1-10 (Remote/Values/Soft Skills)>
          }},
          "comparisons": [
            {{
                "category": "🧗 Level",
                "me_val": "<My Level>",
                "them_val": "<Job Level>",
                "icon": "<✅/➖/❌/⚠️>"
            }},
            {{
                "category": "🏠 Type",
                "me_val": "<My Type>",
                "them_val": "<Job Type>",
                "icon": "<✅/➖/❌/⚠️>"
            }}
          ],
          "tech_stack": {{
              "verdict": "<Short enthusiastic summary>",
              "matches": ["<Tool 1>", "<Tool 2>", "<Tool 3>", "<Tool 4>", "<Tool 5>"],
              "gaps": ["<Gap 1>", "<Gap 2>", "<Gap 3>"]
          }},
          "one_line_summary": "<One sentence summary>"
        }}
        """

        try:
            message = self.client.messages.create(
                model=self.model,
                max_tokens=800,
                system=system,
                messages=[{"role": "user", "content": prompt}],
            )

            response_text = message.content[0].text.strip()
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0]
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0]

            data = json.loads(response_text.strip())

            storage_payload = {
                "comparisons": data.get("comparisons", []),
                "tech_stack": data.get("tech_stack", {}),
                "summary": data.get("one_line_summary", "No summary provided"),
            }

            # Calculate Average Score Python Side
            scores = data.get("match_scores", {})
            if scores:
                avg_score = sum(scores.values()) / len(scores)
            else:
                avg_score = 0

            return {
                "job_id": job["id"],
                "match_scores": json.dumps(scores),
                "avg_score": avg_score,
                "reasoning": json.dumps(storage_payload),
            }

        except Exception as e:
            print(f"Error evaluating job {job.get('id')}: {e}")
            return {
                "job_id": job["id"],
                "match_scores": "{}",
                "avg_score": 0,
                "reasoning": json.dumps({"summary": "Error parsing response"}),
            }
