import json
import re
import threading
import requests
import uuid

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_TAGS_URL = "http://localhost:11434/api/tags"
MODEL_NAME = "llama3.2:3b"


def classify_question(question):
    q = question.lower().strip()

    if re.search(
        r"\b(difference between|differentiate|compare|comparison|versus|vs)\b",
        q
    ):
        return "comparison"

    if re.search(
        r"\b(explain|describe|how does|how do|how is|why does|why do|"
        r"how can|steps involved|working of)\b",
        q
    ):
        return "explanation"

    if re.search(
        r"\b\d+\s*[\+\-\*/x×÷]\s*\d+",
        q
    ):
        return "calculation"

    if re.search(
        r"\b(capital of|invented|created|founded|largest|smallest|"
        r"fastest|tallest|oldest|who is|who was|when was|where is|"
        r"which data structure|how many|how much)\b",
        q
    ):
        return "fact"

    if re.search(r"^(what is|what are|define|meaning of)\b", q):
        return "definition"

    return "general"


class _Job:
    def __init__(self, job_id):
        self.job_id = job_id
        self.cancel_event = threading.Event()
        self.response = None
        self.response_lock = threading.Lock()


class LLMEngine:

    def __init__(self):
        self.current_job = None
        self.job_lock = threading.Lock()

    def check_server(self):
        try:
            response = requests.get(
                OLLAMA_TAGS_URL,
                timeout=5
            )
            response.raise_for_status()

            models = response.json().get("models", [])

            available = [
                model.get("name", "")
                for model in models
            ]

            model_found = any(
                name == MODEL_NAME or name.startswith(MODEL_NAME + ":")
                for name in available
            )

            if not model_found:
                return False, (
                    f"Model {MODEL_NAME} not found. "
                    f"Run: ollama pull {MODEL_NAME}"
                )

            return True, f"Ollama ready ({MODEL_NAME})"

        except requests.RequestException as error:
            return False, f"Ollama is not running: {error}"

        except Exception as error:
            return False, f"Ollama check failed: {error}"

    def warm_up(self):
        payload = {
            "model": MODEL_NAME,
            "prompt": "Reply with OK.",
            "stream": False,
            "keep_alive": "30m",
            "options": {
                "num_predict": 2,
                "temperature": 0
            }
        }

        try:
            requests.post(
                OLLAMA_URL,
                json=payload,
                timeout=(5, 120)
            ).raise_for_status()
            return True

        except requests.RequestException:
            return False

    def cancel(self):
        with self.job_lock:
            job = self.current_job
            self.current_job = None

        if job:
            job.cancel_event.set()

            with job.response_lock:
                if job.response is not None:
                    try:
                        job.response.close()
                    except Exception:
                        pass

    def start_answer(self, question, on_token, on_done, on_error):
        self.cancel()

        job_id = str(uuid.uuid4())
        job = _Job(job_id)

        with self.job_lock:
            self.current_job = job

        thread = threading.Thread(
            target=self._run,
            args=(job, question, on_token, on_done, on_error),
            daemon=True
        )

        thread.start()

        return job_id

    def _run(self, job, question, on_token, on_done, on_error):

        answer_type = classify_question(question)

        instructions = {
            "fact": (
                "Give only the exact factual answer. "
                "Use one word or a short phrase whenever possible."
            ),

            "calculation": (
                "Calculate carefully and return only the final result. "
                "Do not show unnecessary working."
            ),

            "definition": (
                "Give a clear definition in 2 to 3 sentences. "
                "Explain its meaning and main purpose."
            ),

            "explanation": (
                "Explain the concept in a clear, informative paragraph. "
                "Include how it works and important details."
            ),

            "comparison": (
                "Compare the concepts using concise bullet points. "
                "Include the important differences."
            ),

            "general": (
                "Give a direct and accurate answer with an appropriate "
                "amount of detail."
            )
        }

        token_limits = {
            "fact": 20,
            "calculation": 16,
            "definition": 100,
            "explanation": 160,
            "comparison": 180,
            "general": 110
        }

        prompt = f"""
You are a fast and accurate technical interview assistant.

Answer the exact question asked.

Instructions:
{instructions[answer_type]}

Rules:
1. Never repeat the question.
2. Never start with "The answer is".
3. Do not add greetings or introductions.
4. Do not invent facts.
5. Use simple and correct technical language.
6. Keep the response relevant.
7. For simple factual questions, answer briefly.
8. For technical explanations, provide enough detail.
9. Do not unnecessarily expand short answers.

Question: {question}

Answer:
"""

        payload = {
            "model": MODEL_NAME,
            "prompt": prompt,
            "stream": True,
            "keep_alive": "30m",
            "options": {
                "temperature": 0,
                "num_predict": token_limits[answer_type],
                "num_ctx": 2048,
                "top_k": 10,
                "top_p": 0.8
            }
        }

        response = None

        try:
            if job.cancel_event.is_set():
                return

            response = requests.post(
                OLLAMA_URL,
                json=payload,
                stream=True,
                timeout=(5, 120)
            )

            with job.response_lock:
                job.response = response

            response.raise_for_status()

            for line in response.iter_lines(chunk_size=1):

                if job.cancel_event.is_set():
                    return

                if not line:
                    continue

                try:
                    data = json.loads(line.decode("utf-8"))
                except (json.JSONDecodeError, UnicodeDecodeError):
                    continue

                token = data.get("response", "")

                if token and not job.cancel_event.is_set():
                    on_token(job.job_id, token)

                if data.get("done"):
                    break

            if not job.cancel_event.is_set():
                on_done(job.job_id)

        except requests.RequestException as error:
            if not job.cancel_event.is_set():
                on_error(
                    job.job_id,
                    f"Ollama connection error: {error}"
                )

        except Exception as error:
            if not job.cancel_event.is_set():
                on_error(
                    job.job_id,
                    f"Answer generation failed: {error}"
                )

        finally:
            with job.response_lock:
                if job.response is not None:
                    try:
                        job.response.close()
                    except Exception:
                        pass

                    job.response = None

            with self.job_lock:
                if self.current_job is job:
                    self.current_job = None