import asyncio
import json
import os
import time
from typing import Dict, Any, List

from google import genai

from app import config
from app.core.agent_service import agent_service
from app.utils.logger import get_logger

logger = get_logger("arc_fusion.evaluation_service")


class EvaluationService:
    def __init__(self):
        self.client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
        self.evaluation_model = config.EVALUATION_MODEL
        self.golden_dataset_path = "data/golden_dataset.jsonl"
        self.rate_limit_delay = 2  # Increased delay for "pro" model

    def _load_golden_dataset(self) -> List[Dict[str, str]]:
        if not os.path.exists(self.golden_dataset_path):
            logger.error(f"Golden dataset not found at: {self.golden_dataset_path}")
            return []
        with open(self.golden_dataset_path, "r") as f:
            return [json.loads(line) for line in f]

    async def _evaluate_single_item(self, item: Dict[str, str]) -> Dict[str, Any]:
        question = item["question"]
        ground_truth = item["ground_truth"]

        # Get RAG response
        rag_response = await agent_service.process_query(query=question)
        generated_answer = rag_response.get("answer", "")

        # Evaluate with Gemini
        prompt = self._create_evaluation_prompt(
            question, ground_truth, generated_answer
        )
        try:
            eval_response = self.client.models.generate_content(
                model=self.evaluation_model, contents=prompt
            )
            eval_result = self._parse_evaluation_response(eval_response.text)
        except Exception as e:
            logger.error(f"Error evaluating with Gemini: {e}")
            eval_result = {"error": str(e)}

        return {
            "question": question,
            "ground_truth": ground_truth,
            "generated_answer": generated_answer,
            "evaluation": eval_result,
            "citations": rag_response.get("citations", []),
        }

    async def run_evaluation(self) -> Dict[str, Any]:
        dataset = self._load_golden_dataset()
        if not dataset:
            return {"error": "Golden dataset is empty or not found."}

        results = []
        for item in dataset:
            result = await self._evaluate_single_item(item)
            results.append(result)
            await asyncio.sleep(self.rate_limit_delay)

        # Aggregate results
        summary = self._summarize_results(results)

        return {"summary": summary, "results": results}

    def _create_evaluation_prompt(
        self, question: str, ground_truth: str, generated_answer: str
    ) -> str:
        return f"""
You are an expert evaluator for a RAG system. Your task is to assess the quality of a generated answer based on a ground truth answer.

Please evaluate the generated answer based on the following criteria:
1.  **Faithfulness**: Does the answer stay true to the context provided in the ground truth?
2.  **Answer Relevancy**: Is the answer relevant to the question?
3.  **Completeness**: Does the answer cover all aspects of the ground truth?

Question: "{question}"
Ground Truth: "{ground_truth}"
Generated Answer: "{generated_answer}"

Provide a score from 1 to 5 for each criterion (1=Poor, 5=Excellent) and a brief reasoning for each score.

Respond in JSON format:
{{
  "faithfulness": {{ "score": <score>, "reasoning": "<text>" }},
  "answer_relevancy": {{ "score": <score>, "reasoning": "<text>" }},
  "completeness": {{ "score": <score>, "reasoning": "<text>" }}
}}
"""

    def _parse_evaluation_response(self, response_text: str) -> Dict[str, Any]:
        try:
            # Clean up the response text before parsing
            clean_text = (
                response_text.strip().replace("```json", "").replace("```", "").strip()
            )
            return json.loads(clean_text)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse evaluation JSON: {e}")
            return {"error": "Failed to parse evaluation response", "raw": response_text}

    def _summarize_results(self, results: List[Dict[str, Any]]) -> Dict[str, float]:
        total_items = len(results)
        if total_items == 0:
            return {}

        avg_faithfulness = (
            sum(
                r["evaluation"].get("faithfulness", {}).get("score", 0)
                for r in results
                if "error" not in r["evaluation"]
            )
            / total_items
        )
        avg_relevancy = (
            sum(
                r["evaluation"].get("answer_relevancy", {}).get("score", 0)
                for r in results
                if "error" not in r["evaluation"]
            )
            / total_items
        )
        avg_completeness = (
            sum(
                r["evaluation"].get("completeness", {}).get("score", 0)
                for r in results
                if "error" not in r["evaluation"]
            )
            / total_items
        )

        return {
            "average_faithfulness": round(avg_faithfulness, 2),
            "average_answer_relevancy": round(avg_relevancy, 2),
            "average_completeness": round(avg_completeness, 2),
            "total_evaluated": total_items,
        }


evaluation_service = EvaluationService() 