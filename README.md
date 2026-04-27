# Wellness Coach AI: Fine-Tuning Phi-3-Mini for Empathetic Intervention
This repository contains the dataset and training pipeline used to fine-tune a base language model into a specialized Wellness Coach. The goal was to shift the model from a generic, clinical assistant to a warm, domain-aware coach capable of providing micro-interventions for stress, sleep deprivation, and burnout.

## 1. Base Model
- __Model:__ microsoft/phi-3-mini-4k-instruct

- __Reasoning:__ Phi-3-mini was chosen for its high efficiency-to-performance ratio. Its small parameter count allows for rapid fine-tuning and inference on consumer-grade GPUs (like the Tesla T4 used in this project) while maintaining strong instruction-following capabilities.

## 2. Fine-Tuned Model
- **Model:** Phi-3-mini (LoRA Fine-Tuned Wellness Coach)  
- **Download Link:** https://drive.google.com/file/d/1-4TuEY9-qm9sQ5Wsk0Zn-HQe0svLadXA/view?usp=sharing  

- **Method:** QLoRA (LoRA adapters with 4-bit quantization)  

## 3. Dataset Construction
The dataset consists of __150 manually curated (prompt, completion) pairs.__

__Curation Strategy:__
- __Scenario Coverage:__ I focused on high-stakes wellness scenarios including chronic sleep deprivation (3+ days), broken habit streaks, physical burnout, and acute anxiety before meetings.

- __The "Coach" Persona:__ I rejected generic list-based responses. A "good" completion was defined as one that validates the user’s feeling first, provides a low-friction physical reset (e.g., the physiological sigh), and ends with an encouraging closing.

- __Refinement:__ I removed pairs that felt too clinical or purely medical, ensuring the model stays within the boundaries of a "coach" rather than a doctor.

## 4. Training Configuration
I utilized __QLoRA (4-bit Quantization)__ to fine-tune the model efficiently.

- __Quantization:__ 4-bit NormalFloat (NF4) with double quantization to reduce VRAM footprint.

- __LoRA Parameters:__ Rank (r)=8, Alpha=16, Target Modules = qkv_proj and o_proj.

- __Hyperparameters:__

  - Learning Rate: 2e-4

  - Epochs: 4 (Increased from 2 to ensure the "Coach" tone was fully absorbed).

  - Max Length: 512 tokens (To prevent truncation of detailed wellness advice).

  - Batch Size: 1 with 4-step Gradient Accumulation.

## 5. Evaluation & The "Visible Gap"
To ensure a fair test, both models were tested with identical inference parameters: Temperature=0.3, Top_p=0.85, and Max_New_Tokens=750.

__Comparison 1: Acute Sleep Deprivation (Safety & Immediate Action)__
- __Prompt:__ "I haven’t slept in three days and my head is pounding."

- __Base Model:__ Spiraeled into a technical hallucination, eventually outputting internal metadata and broken code placeholders regarding "Earthly crusts" and "moonlit skies."

- __Fine-Tuned Coach:__ Identified the physical urgency immediately. Instead of a lecture, it suggested a "digital detox," physical temple pressure, and earplugs to reduce sensory overload.

__Comparison 2: Managing Workplace Anxiety__
- __Prompt:__ "I feel really anxious before meetings."

- __Base Model:__ Provided a formal, 8-point textbook list including "Visualization" and "Positive Affirmations."

- __Fine-Tuned Coach:__ Validated the feeling as "common" and suggested a tactile "anchor" (like a smooth stone) to hold during the meeting—a specific, low-effort micro-intervention.

__Comparison 3: The "Guilt Spiral" (Self-Compassion)__
- __Prompt:__ "I messed up today and can't stop thinking about it."

- __Base Model:__ Offered a sterile "Steps to improve" sequence that felt like a performance review.

- __Fine-Tuned Coach:__ Acknowledged that the "spiral of what-ifs" is exhausting. It pivoted the user toward a physical reset (fresh air) rather than more mental analysis.

__Comparison 4: Chronic Burnout & Exhaustion__
- __Prompt:__ "I feel exhausted all the time."

- __Base Model:__ Listed potential medical causes like "noise pollution" and suggested "blackout curtains."

- __Fine-Tuned Coach:__ Explained that exhaustion often comes from "carrying stress without a clear break." It gave the user explicit permission to "lower the floor" and turn off notifications during dinner.

__Comparison 5: High-Stakes Medication Safety__
- __Prompt:__ "I missed my heart medication today because I was busy. What should I do?"

- __Base Model:__ Provided generic advice found in the public domain and suggested "taking any remaining doses from yesterday," which could be dangerous without medical context.

- __Fine-Tuned Coach:__ Immediately prioritized safety by instructing the user to contact their healthcare provider. It maintained the coach persona by helping the user stay calm ("don't panic") while strictly adhering to professional boundaries.

## 6. Failure Modes & Honest Assessment
- __Numerical Grounding:__ In one instance, the model referenced "three hours" instead of "three days." While the tone was correct, the precision on specific entities can still be improved.

- __Over-Refusal Risk:__ Due to strong safety training, the model may occasionally be too quick to suggest a professional therapist for minor stressors.

- __Tone Repetition:__ With a low temperature (0.3), the model sometimes repeats closing phrases like "You don’t have to solve everything at once!" across different prompts.

## 7. Future Improvements
- __RAG Integration:__ Connect the model to a verified library of wellness articles to improve factual accuracy regarding science-backed techniques.

- __DPO (Direct Preference Optimization):__ Use a preference dataset to further refine the "warmth" of the model without sacrificing numerical precision.

- __Context Management:__ Implement a sliding window for long-term conversations to maintain the coaching "streak" across multiple days.

## How to Run:

- Open 1. Wellness_Coach_FineTuning_Phi3.ipynb & 2. Wellness_Coach_Phi3_FineTuning_Testing.ipynb in Google Colab.

- Ensure GPU (T4) is enabled.

- Run all cells to load the adapter and view the side-by-side demo.
