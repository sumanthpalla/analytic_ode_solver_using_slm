import sympy
import random
import json

# Initialize SymPy variables
x, y = sympy.symbols('x y')
u = sympy.Function('u')(x) # Dummy variable for substitution in Homogeneous
C = sympy.Symbol('C') # Constant for random coefficients

# --- Import Configuration ---

import os
import yaml

with open(os.path.join(os.path.dirname(__file__), "..", "..", "config", "data_generation_config.yaml"), "r") as f:
    config = yaml.safe_load(f)


NUM_SAMPLES_PER_CLASS = config["num_samples_per_class"]
ODE_CLASSES = config["ode_classes"]
OUTPUT_FILE = config["output_file"]


# ---------------------

def get_random_function(var):
    """Generates a random SymPy function of a single variable."""
    funcs = [
        sympy.sin(var),
        sympy.cos(var),
        sympy.exp(var),
        sympy.log(sympy.Abs(var)),
        var**2,
        var**3,
        sympy.sqrt(var)
    ]
    # Mix simple functions and polynomial terms
    poly = sum(random.choice([-3, -2, -1, 1, 2, 3]) * var**p for p in range(1, 4))
    return random.choice(funcs) + poly

def generate_random_coefficient():
    """Generates a random SymPy coefficient (integer or simple variable term)."""
    return random.choice([-5, -3, -2, 2, 3, 5]) * random.choice([1, x, x**2, y, y**2])

# --- ODE Generation Functions ---

def generate_separable():
    """Generates an ODE of the form g(y) dy/dx = f(x)"""
    f_x = get_random_function(x)
    g_y = get_random_function(y)
    ode_expr = f_x / g_y # dy/dx = f(x)/g(y)

    return {
        "ode": sympy.Eq(sympy.Derivative(y, x), ode_expr),
        "classification": "Separable ODE",
        "hint": "The equation can be rearranged to the form $g(y)dy = f(x)dx$. Integrate both sides."
    }

def generate_linear():
    """Generates an ODE of the form dy/dx + P(x)y = Q(x)"""
    P_x = get_random_function(x) + random.choice([0, C*x])
    Q_x = get_random_function(x) + random.choice([0, C*x**2])

    # ODE: dy/dx = Q(x) - P(x)*y
    ode_expr = Q_x - P_x * y

    return {
        "ode": sympy.Eq(sympy.Derivative(y, x), ode_expr),
        "classification": "Linear First-Order ODE",
        "hint": "Calculate the integrating factor $\mu(x) = e^{\int P(x) dx}$ and multiply the equation by it."
    }

def generate_exact():
    """Generates an Exact ODE where dM/dy = dN/dx"""
    # Start with a potential function f(x, y)
    f_xy = get_random_function(x) * get_random_function(y) + x**2 * y + y**3 * x
    M = sympy.diff(f_xy, x) # M = d/dx f(x,y)
    N = sympy.diff(f_xy, y) # N = d/dy f(x,y)

    # ODE: M(x,y) dx + N(x,y) dy = 0. Solve for dy/dx: dy/dx = -M/N
    ode_expr = -M / N

    return {
        "ode": sympy.Eq(sympy.Derivative(y, x), ode_expr.simplify()),
        "classification": "Exact ODE",
        "hint": "Verify exactness by checking $\partial M/\partial y = \partial N/\partial x$. The solution is $f(x,y) = C$ where $\partial f/\partial x = M$ and $\partial f/\partial y = N$."
    }

def generate_homogeneous():
    """Generates an ODE of the form dy/dx = f(y/x)"""
    # f(v) where v = y/x
    v = y/x
    f_v = get_random_function(v) + random.choice([v, v**2])

    # ODE: dy/dx = f(y/x)
    ode_expr = f_v

    return {
        "ode": sympy.Eq(sympy.Derivative(y, x), ode_expr),
        "classification": "Homogeneous ODE",
        "hint": "Use the substitution $y = vx$, which implies $\frac{dy}{dx} = v + x \frac{dv}{dx}$. The resulting equation will be separable."
    }

def generate_bernoulli():
    """Generates an ODE of the form dy/dx + P(x)y = Q(x)y^n, where n != 0, 1"""
    P_x = get_random_function(x)
    Q_x = get_random_function(x)
    n = random.choice([2, 3, 4]) # Exponent n, must be != 0, 1

    # ODE: dy/dx = Q(x)y^n - P(x)y
    ode_expr = Q_x * y**n - P_x * y

    return {
        "ode": sympy.Eq(sympy.Derivative(y, x), ode_expr),
        "classification": "Bernoulli ODE",
        "hint": f"Use the substitution $u = y^{1-n}$ (or $u = y^{1-{n}}$ in LaTeX), which transforms the equation into a linear ODE in $u$."
    }

# --- Main Generation Loop ---

def generate_ode_dataset(num_samples_per_class):
    """Generates the full dataset by sampling from all generators."""
    generators = [
        generate_separable,
        generate_linear,
        generate_exact,
        generate_homogeneous,
        generate_bernoulli
    ]

    dataset = []

    print(f"Generating {len(generators) * num_samples_per_class} ODE samples...")

    for generator in generators:
        for i in range(num_samples_per_class):
            try:
                sample = generator()

                # Format the ODE using SymPy's LaTeX printer
                # Use str() instead of latex() for a simpler text-based prompt
                ode_latex = sympy.latex(sample["ode"])

                # Create the final training example structure
                training_example = {
                    "input": f"Classify the following ODE and provide the first step for the analytic solution. ODE: ${ode_latex}$",
                    "output": json.dumps({
                        "classification": sample["classification"],
                        "hint": sample["hint"]
                    })
                }

                dataset.append(training_example)

            except Exception as e:
                # Handle cases where SymPy cannot simplify or process the expression
                # print(f"Skipped sample due to error: {e}")
                continue

    print(f"Successfully generated {len(dataset)} samples.")
    return dataset

# --- Execution ---

if __name__ == "__main__":
    ode_data = generate_ode_dataset(NUM_SAMPLES_PER_CLASS)

    # Shuffle the data to mix classes before saving
    random.shuffle(ode_data)

    # Save the dataset to a JSON file in data/raw_data
    base_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.normpath(os.path.join(base_dir, "..", "..", "data", "raw_data", OUTPUT_FILE))

    # 2. Ensure the directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # 3. Save the dataset using the full save_path
    with open(save_path, 'w') as f:
        json.dump(ode_data, f, indent=2)

    print(f"File saved successfully to: {save_path}")
    print(f"Data saved to {save_path}. Ready for fine-tuning!")

    # Print the first few samples for verification
    print("\n--- Sample Data Verification ---")
    for i in range(min(5, len(ode_data))):
        print(f"--- Sample {i+1} ---")
        print(f"Input: {ode_data[i]['input']}")
        # Use json.loads for clean printing of the output
        print(f"Output: {json.loads(ode_data[i]['output'])}")