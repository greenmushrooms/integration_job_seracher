import json
import os
from typing import Dict, List

import anthropic
import pandas as pd


class ClaudeJobEvaluator:
    def __init__(self, model: str = "claude-3-5-haiku-20241022"):
        self.client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        self.model = model

        # Define the schema strictly as a Tool
        self.tool_schema = {
            "name": "submit_job_evaluation",
            "description": "Submit the evaluation of a job posting against a candidate resume.",
            "input_schema": {
                "type": "object",
                "properties": {
                    # MOVED TECH ANALYSIS TO THE TOP
                    "tech_stack_analysis": {
                        "type": "object",
                        "properties": {
                            "verdict": {"type": "string"},
                            "matches": {"type": "array", "items": {"type": "string"}},
                            # Force it to list at least 1 gap if score < 10
                            "gaps": {"type": "array", "items": {"type": "string"}},
                        },
                        "required": ["verdict", "matches", "gaps"],
                    },
                    "verdict": {
                        "type": "string",
                        "enum": ["Step Up", "Lateral", "Title Regression", "Pivot"],
                    },
                    # SCORES COME LAST
                    "match_scores": {
                        "type": "object",
                        "properties": {
                            "skills_match": {
                                "type": "integer",
                                "description": "10=Perfect, 8=Strong, 6=Ramp-up needed. Penalize for domain switches.",
                                "minimum": 1,
                                "maximum": 10,
                            },
                            "career_level_alignment": {
                                "type": "integer",
                                "minimum": 1,
                                "maximum": 10,
                            },
                            "experience_relevance": {
                                "type": "integer",
                                "minimum": 1,
                                "maximum": 10,
                            },
                            "culture_fit": {
                                "type": "integer",
                                "minimum": 1,
                                "maximum": 10,
                            },
                        },
                        "required": [
                            "skills_match",
                            "career_level_alignment",
                            "experience_relevance",
                            "culture_fit",
                        ],
                    },
                    "one_line_summary": {"type": "string"},
                },
                # Enforce the order in the prompt logic (though JSON is unordered,
                # LLMs tend to generate in definition order or you can enforce it via prompt)
                "required": [
                    "tech_stack_analysis",  # Analysis first
                    "verdict",
                    "match_scores",  # Score last
                    "one_line_summary",
                ],
            },
        }

    def evaluate(self, resume: str, jobs_df: pd.DataFrame) -> pd.DataFrame:
        results = []

        # System Prompt: Clean and persona-driven
        system_text = """You are an expert Talent Evaluator.
        Your goal is to identify growth opportunities.
        1. Analyze the candidate's RESUME to establish a baseline.
        2. Compare strictly against the JOB DESCRIPTION."""

        # Cache the resume (Ephemeral caching)
        system_messages = [
            {"type": "text", "text": system_text},
            {
                "type": "text",
                "text": f"<candidate_resume>\n{resume}\n</candidate_resume>",
                "cache_control": {"type": "ephemeral"},
            },
        ]

        print(f"Starting evaluation of {len(jobs_df)} jobs...")

        for idx, job in jobs_df.iterrows():
            print(
                f"Evaluating {idx + 1}/{len(jobs_df)}: {job.get('company', 'Unknown')}"
            )

            # Error handling wrapper
            try:
                result = self._score_job_with_tool(job, system_messages)
                results.append(result)
            except Exception as e:
                print(f"FAILED on job {job.get('id', 'unknown')}: {e}")
                results.append(
                    {
                        "job_id": job.get("id"),
                        "avg_score": 0,
                        "match_scores": "{}",
                        "reasoning": json.dumps({"summary": f"Error: {str(e)}"}),
                    }
                )

        return pd.DataFrame(results)

    def _score_job_with_tool(self, job: pd.Series, system_messages: List[Dict]) -> Dict:
        # Don't truncate descriptions! Claude handles 200k tokens.
        user_content = f"""
        Please evaluate this position:

        COMPANY: {job.get("company", "Unknown")}
        TITLE: {job.get("title", "Unknown")}

        DESCRIPTION:
        {job.get("description", "")}
        """

        message = self.client.messages.create(
            model=self.model,
            max_tokens=2000,
            temperature=0,
            system=system_messages,
            messages=[{"role": "user", "content": user_content}],
            tools=[self.tool_schema],
            tool_choice={
                "type": "tool",
                "name": "submit_job_evaluation",
            },
        )

        # Extract the JSON from the tool use input
        tool_use = next(block for block in message.content if block.type == "tool_use")
        data = tool_use.input

        # Calculate Average Score
        scores = data.get("match_scores", {})
        avg_score = sum(scores.values()) / len(scores) if scores else 0

        # DATA PACKING FOR DB COMPATIBILITY
        # We pack the new fields into 'reasoning' so they fit into the existing JSON column.
        # We map keys to match what helper.py expects (tech_stack, summary).
        reasoning_payload = {
            "verdict": data.get("verdict"),
            "tech_stack": data.get("tech_stack_analysis"),
            "summary": data.get("one_line_summary"),
            # Comparisons are not generated by the new prompt, so we pass empty list
            # to prevent helper.py from breaking.
            "comparisons": [],
        }

        return {
            "job_id": job.get("id"),
            "avg_score": avg_score,
            "match_scores": json.dumps(scores),
            "reasoning": json.dumps(reasoning_payload),
        }
