import json
import os
from typing import Dict, List

import anthropic
import pandas as pd
from prefect.blocks.system import Secret

DB_HOST = Secret.load("job-searcher--anthropic-api-key").get()


class ClaudeJobEvaluator:
    """Evaluates job postings against a resume using Claude API"""

    def __init__(self, model: str = "claude-haiku-4-5"):
        self.client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        self.model = model

    def evaluate(self, resume: str, jobs_df: pd.DataFrame) -> pd.DataFrame:
        """
        Evaluate multiple jobs against a resume.

        Args:
            resume: Candidate's resume text
            jobs_df: DataFrame with job descriptions (must have 'id', 'company', 'description', 'title')

        Returns:
            DataFrame with job_id, match_scores (JSON), total_score, and reasoning
        """
        results = []

        # Build system prompt with cached resume
        system = [
            {
                "type": "text",
                "text": """You are a job-resume matching expert. Analyze jobs against the provided resume and score on these 5 metrics (1-10 scale):
1. skills_match - Technical skills alignment
2. experience_relevance - Role/responsibility alignment
3. keywords_ats - ATS keyword optimization
4. career_level_alignment - Seniority/progression fit
5. soft_skills_cultural_fit - Leadership/culture alignment

Always respond with valid JSON only. NO markdown, NO backticks, NO explanatory text outside the JSON.""",
            },
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
        """Score a single job against the cached resume"""

        prompt = f"""
JOB DESCRIPTION:
Company: {job.get("company", "Not specified")}
Title: {job.get("title", "Not specified")}

{job.get("description", "No description available")}

CRITICAL INSTRUCTIONS:
- Respond with ONLY the JSON object below
- DO NOT use markdown code blocks
- DO NOT use backticks
- DO NOT include any text before or after the JSON
- The entire response must be valid, parseable JSON

{{
  "skills_match": <1-10>,
  "experience_relevance": <1-10>,
  "keywords_ats": <1-10>,
  "career_level_alignment": <1-10>,
  "soft_skills_cultural_fit": <1-10>,
  "reasoning": "<brief explanation>"
}}
"""

        message = self.client.messages.create(
            model=self.model,
            max_tokens=1000,
            system=system,
            messages=[{"role": "user", "content": prompt}],
        )

        # Parse the JSON response
        response_text = message.content[0].text

        # Clean up potential markdown formatting
        response_text = response_text.strip()
        if response_text.startswith("```json"):
            response_text = response_text[7:]
        if response_text.startswith("```"):
            response_text = response_text[3:]
        if response_text.endswith("```"):
            response_text = response_text[:-3]
        response_text = response_text.strip()

        scores = json.loads(response_text)
        reasoning = scores.pop("reasoning", "No reasoning provided")
        avg_score = sum(scores.values()) / len(scores)

        return {
            "job_id": job["id"],
            "match_scores": json.dumps(scores),
            "avg_score": avg_score,
            "reasoning": reasoning,
        }
