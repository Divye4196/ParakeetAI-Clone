import customtkinter as ctk
import threading

from modules.audio_capture import (
    start_recording,
    stop_recording,
    get_audio_chunk
)

from modules.speech_to_text import transcribe_chunk

from modules.llm_engine import (
    detect_question,
    generate_answer
)


ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


class Dashboard(ctk.CTk):

    def __init__(self):
        super().__init__()

        self.title("Parakeet AI Assistant")
        self.geometry("950x750")
        self.resizable(False, False)

        self.question = ""
        self.is_recording = False
        self.transcription_parts = []
        self.detected_questions = set()

        self.title_label = ctk.CTkLabel(
            self,
            text="PARAKEET AI ASSISTANT",
            font=("Arial", 30, "bold")
        )
        self.title_label.pack(pady=25)

        self.status_label = ctk.CTkLabel(
            self,
            text="Status: Ready",
            font=("Arial", 14)
        )
        self.status_label.pack(pady=10)

        self.button_frame = ctk.CTkFrame(
            self,
            fg_color="transparent"
        )
        self.button_frame.pack(pady=15)

        self.start_button = ctk.CTkButton(
            self.button_frame,
            text="Start Recording",
            command=self.start_recording,
            width=180,
            height=45
        )
        self.start_button.pack(side="left", padx=10)

        self.stop_button = ctk.CTkButton(
            self.button_frame,
            text="Stop Recording",
            command=self.stop_recording,
            width=180,
            height=45,
            state="disabled",
            fg_color="#C0392B",
            hover_color="#922B21"
        )
        self.stop_button.pack(side="left", padx=10)

        self.transcription_label = ctk.CTkLabel(
            self,
            text="Live Transcription",
            font=("Arial", 20, "bold")
        )
        self.transcription_label.pack(pady=10)

        self.transcription_box = ctk.CTkTextbox(
            self,
            width=850,
            height=150,
            font=("Arial", 15)
        )
        self.transcription_box.pack(pady=10)

        self.answer_label = ctk.CTkLabel(
            self,
            text="Automatically Suggested Answer",
            font=("Arial", 20, "bold")
        )
        self.answer_label.pack(pady=10)

        self.answer_box = ctk.CTkTextbox(
            self,
            width=850,
            height=200,
            font=("Arial", 15)
        )
        self.answer_box.pack(pady=10)

    # START RECORDING

    def start_recording(self):

        try:

            start_recording()

            self.is_recording = True
            self.transcription_parts = []
            self.detected_questions = set()
            self.question = ""

            self.transcription_box.delete("1.0", "end")
            self.answer_box.delete("1.0", "end")

            self.start_button.configure(state="disabled")
            self.stop_button.configure(state="normal")

            self.status_label.configure(
                text="Status: Listening..."
            )

            threading.Thread(
                target=self.live_transcription_worker,
                daemon=True
            ).start()

        except Exception as e:

            self.status_label.configure(
                text=f"Recording Error: {str(e)}"
            )

            self.start_button.configure(state="normal")

    # LIVE TRANSCRIPTION

    def live_transcription_worker(self):

        while True:

            chunk = get_audio_chunk()

            if chunk is None:
                continue

            if isinstance(chunk, str) and chunk == "STOP":
                break

            try:

                transcription = transcribe_chunk(chunk)

                if transcription:

                    self.transcription_parts.append(
                        transcription
                    )

                    self.question = " ".join(
                        self.transcription_parts
                    )

                    self.after(
                        0,
                        self.update_live_transcription
                    )

                    threading.Thread(
                        target=self.check_for_question,
                        args=(self.question,),
                        daemon=True
                    ).start()

            except Exception as e:

                print("TRANSCRIPTION ERROR:", str(e))

        self.after(
            0,
            lambda: self.status_label.configure(
                text="Status: Recording stopped"
            )
        )

    # UPDATE TRANSCRIPTION BOX

    def update_live_transcription(self):

        self.transcription_box.delete("1.0", "end")

        self.transcription_box.insert(
            "1.0",
            self.question
        )

    # STOP RECORDING

    def stop_recording(self):

        if not self.is_recording:
            return

        self.is_recording = False

        self.start_button.configure(state="disabled")
        self.stop_button.configure(state="disabled")

        self.status_label.configure(
            text="Status: Finishing transcription..."
        )

        try:

            audio_path = stop_recording()

            if not audio_path:

                self.status_label.configure(
                    text="No audio recorded."
                )

            self.start_button.configure(state="normal")

        except Exception as e:

            self.status_label.configure(
                text=f"Recording Error: {str(e)}"
            )

            self.start_button.configure(state="normal")

    # AUTOMATIC QUESTION DETECTION

    def check_for_question(self, text):

        try:

            detected = detect_question(text)

            if not detected:
                return

            if detected in self.detected_questions:
                return

            self.detected_questions.add(detected)

            print("QUESTION DETECTED:", detected)

            self.after(
                0,
                lambda q=detected: self.display_detected_question(q)
            )

            self.generate_automatic_answer(detected)

        except Exception as e:

            print("QUESTION DETECTION ERROR:", str(e))

    # DISPLAY DETECTED QUESTION

    def display_detected_question(self, question):

        self.status_label.configure(
            text=f"Question detected: {question}"
        )

        self.answer_box.delete("1.0", "end")

        self.answer_box.insert(
            "1.0",
            f"Detected Question:\n{question}\n\nGenerating answer..."
        )

    # AUTOMATIC ANSWER GENERATION

    def generate_automatic_answer(self, question):

        try:

            self.after(
                0,
                lambda: self.status_label.configure(
                    text="Status: Generating automatic answer..."
                )
            )

            self.after(
                0,
                lambda: self.answer_box.delete("1.0", "end")
            )

            self.after(
                0,
                lambda: self.answer_box.insert(
                    "1.0",
                    f"Question: {question}\n\nAI Answer:\n"
                )
            )

            for token in generate_answer(question):

                self.after(
                    0,
                    lambda t=token: self.answer_box.insert("end", t)
                )

            self.after(
                0,
                lambda: self.status_label.configure(
                    text="Status: Automatic answer ready"
                )
            )

        except Exception as e:

            error_message = str(e)

            print("ANSWER GENERATION ERROR:", error_message)

            self.after(
                0,
                lambda msg=error_message: self.status_label.configure(
                    text=f"AI Error: {msg}"
                )
            )