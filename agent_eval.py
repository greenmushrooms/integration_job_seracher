import json
import os
from typing import Dict, List

import anthropic
import pandas as pd


class ClaudeJobEvaluator:
    """Evaluates job postings against a resume using Claude API"""

    def __init__(self, model: str = "claude-3-haiku-20240307"):
        self.client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        self.model = model

        # UPDATED BASELINE (Using your real info)
        self.current_stats = {
            "compensation": "$120,000 CAD",
            "level": "Lead Data Engineer",
            "stack": "Python, SQL, Azure/Cloud",
            "work_type": "Hybrid",
            "focus": "Data Engineering",
        }

    def evaluate(self, resume: str, jobs_df: pd.DataFrame) -> pd.DataFrame:
        results = []

        # Build system prompt
        system_text = f"""You are a career agent. Compare roles to the candidate's current baseline.

        CANDIDATE BASELINE:
        - Comp: {self.current_stats["compensation"]}
        - Level: {self.current_stats["level"]}

        SCORING (1-10):
        1. skills_match: Stack alignment.
        2. experience_relevance: Problem/Domain fit.
        3. keywords_ats: Keyword density.
        4. career_level_alignment: 10=Higher Pay/Better Title. 5=Lateral. 1=Demotion.
        5. soft_skills: Culture fit.
        """

        system = [
            {"type": "text", "text": system_text},
            {
                "type": "text",
                "text": f"RESUME:\n{resume}",
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
        JOB:
        Company: {job.get("company", "Unknown")}
        Title: {job.get("title", "Unknown")}
        {job.get("description", "")[:4000]}

        INSTRUCTIONS:
        Compare this job to the Baseline. Return JSON.
        For "verdict", use natural language like: "Slightly Higher", "Massive Pay Jump", "Lateral Move", "Lower", "Different Stack".

        {{
          "match_scores": {{
              "skills_match": <1-10>,
              "experience_relevance": <1-10>,
              "keywords_ats": <1-10>,
              "career_level_alignment": <1-10>,
              "soft_skills_cultural_fit": <1-10>
          }},
          "comparisons": [
            {{ "category": "Salary", "verdict": "<e.g. Significantly Higher>", "icon": "<✅/❌/⚠️/➖>" }},
            {{ "category": "Level", "verdict": "<e.g. Step Down>", "icon": "<✅/❌/⚠️/➖>" }},
            {{ "category": "Stack", "verdict": "<e.g. Modernized>", "icon": "<✅/❌/⚠️/➖>" }},
            {{ "category": "Remote", "verdict": "<e.g. Stricter (5 days)>", "icon": "<✅/❌/⚠️/➖>" }}
          ]
        }}
        """

        # Call Claude (Existing code logic...)
        message = self.client.messages.create(
            model=self.model,
            max_tokens=1000,
            system=system,
            messages=[{"role": "user", "content": prompt}],
        )

        # ... (Existing parsing logic) ...
        response_text = message.content[0].text.strip()
        # Clean markdown if present
        if response_text.startswith("```"):
            response_text = response_text.split("```json")[-1].split("```")[0].strip()

        try:
            data = json.loads(response_text)
            scores = data.get("match_scores", {})
            comparisons = data.get("comparisons", [])
            avg = sum(scores.values()) / len(scores) if scores else 0

            return {
                "job_id": job["id"],
                "match_scores": json.dumps(scores),
                "avg_score": avg,
                "reasoning": json.dumps(
                    comparisons
                ),  # Store comparison list as reasoning
            }
        except:
            return {
                "job_id": job["id"],
                "match_scores": "{}",
                "avg_score": 0,
                "reasoning": "[]",
            }
