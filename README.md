# analytic_ode_solver_using_slm
This repo consists of Analytic ODE Solver using Small Language Models


## Fine-tuning & Training Details

### 1. Dataset Curation
The models are trained on a specialized corpus of Ordinary Differential Equations (ODEs) paired with step-by-step analytical solutions.
- **Data Source**: Synthetic generation via SymPy and curated math datasets (e.g., AMPS).
- **Format**: Instruction-based tuning using a `System Prompt -> Hint -> Step-by-Step Solution` structure.
- **Hint Generation**: To improve solver accuracy, the model is trained to first output a "Hint" (identifying the ODE type, e.g., Bernoulli, Exact, or Linear) before generating the integration steps.

### 2. Training Methodology
We utilize Parameter-Efficient Fine-Tuning (PEFT) to adapt SLMs (e.g., Phi-3-mini, TinyLlama) for symbolic manipulation.
- **Framework**: Hugging Face `trl` with `SFTTrainer`.
- **Optimization**: **QLoRA (4-bit quantization)** to minimize VRAM footprint while maintaining 16-bit precision for gradients.
- **Hyperparameters**:
  - **Learning Rate**: 2e-4 with a cosine decay schedule.
  - **LoRA Rank (r)**: 16
  - **LoRA Alpha**: 32
  - **Target Modules**: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`.

### 3. Solving Capability
The training objective focuses on "Chain of Thought" (CoT) prompting for mathematical rigor:
- **Phase 1: Classification**: Identifying the order, degree, and linearity of the ODE.
- **Phase 2: Strategy**: Generating a conceptual hint (e.g., "Substitute $v = y/x$ for homogeneous equations").
- **Phase 3: Execution**: Executing the integration and algebraic simplification to reach the general solution.

### 4. Model Evaluation
The models are evaluated on a set of ODEs with known solutions, focusing on accuracy and efficiency.
- **Metrics**: Accuracy (percentage of correct solutions), efficiency (time to generate solution), and generalization ability.
- **Evaluation**: The models are tested on a diverse set of ODEs, including separable, linear, exact, homogeneous, and Bernoulli equations.

### 5. Future Work
We plan to extend the models to handle more complex ODEs and improve their performance on real-world applications.
- **Future Work**: We plan to extend the models to handle more complex ODEs and improve their performance on real-world applications.



## Model Details

### 1. Model Architecture
The models are based on the Phi-3-mini and TinyLlama architectures, with a focus on parameter efficiency and performance.
- **Phi-3-mini**: A smaller version of the Phi-3 architecture, with a reduced number of parameters.
- **TinyLlama**: A smaller version of the Llama architecture, with a reduced number of parameters.

### 2. Model Training
The models are trained on a set of ODEs with known solutions, focusing on accuracy and efficiency.
- **Metrics**: Accuracy (percentage of correct solutions), efficiency (time to generate solution), and generalization ability.
- **Evaluation**: The models are tested on a diverse set of ODEs, including separable, linear, exact, homogeneous, and Bernoulli equations.

### 3. Model Evaluation
The models are evaluated on a set of ODEs with known solutions, focusing on accuracy and efficiency.
- **Metrics**: Accuracy (percentage of correct solutions), efficiency (time to generate solution), and generalization ability.
- **Evaluation**: The models are tested on a diverse set of ODEs, including separable, linear, exact, homogeneous, and Bernoulli equations.

### 4. Future Work
We plan to extend the models to handle more complex ODEs and improve their performance on real-world applications.
- **Future Work**: We plan to extend the models to handle more complex ODEs and improve their performance on real-world applications.
