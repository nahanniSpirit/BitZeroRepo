"""
BitZero Self-Play Task Generation and Verification

This module implements a simplified task generation and verification system
for the BitZero proof-of-concept, enabling self-play reinforcement learning.
VERSION 3: Corrected math target generation, maintained expanded task variety.
"""

import torch
import numpy as np
import random
import math
import re
import ast # Import ast for syntax checking
import traceback # For more detailed error logging if needed
from typing import Dict, List, Tuple, Optional, Union, Any


class TaskTemplate:
    """Base class for task templates."""
    
    def __init__(self, difficulty: float = 0.5):
        """
        Initialize task template.
        
        Args:
            difficulty: Difficulty level between 0.0 and 1.0
        """
        self.difficulty = max(0.0, min(1.0, difficulty))
        
    def generate(self) -> Tuple[str, Any]:
        """
        Generate a task.
        
        Returns:
            Tuple of (task_description, expected_answer)
        """
        raise NotImplementedError
    
    def verify(self, task: str, answer: str, expected_answer: Any) -> Tuple[bool, float, float]:
        """
        Verify if the answer is correct.
        
        Args:
            task: Task description
            answer: Proposed answer
            expected_answer: The ground truth or specification for the answer.
            
        Returns:
            Tuple of (is_correct, confidence, syntax_reward)
        """
        raise NotImplementedError


class MathTaskTemplate(TaskTemplate):
    """Template for generating mathematical reasoning tasks."""
    
    def __init__(self, difficulty: float = 0.5):
        super().__init__(difficulty)
        
    def generate(self) -> Tuple[str, Any]:
        if self.difficulty < 0.33:
            return self._generate_arithmetic()
        elif self.difficulty < 0.66:
            return self._generate_algebra()
        else:
            return self._generate_word_problem()
    
    def _generate_arithmetic(self) -> Tuple[str, float]:
        """
        Generates an arithmetic task.
        Crucially, the expected_answer is now derived by evaluating the final expression string.
        """
        # Determine the number of operands (and thus operations)
        # More operands for slightly higher difficulty within the arithmetic range
        if self.difficulty < 0.1:
            num_operands = random.randint(2, 3) # Simpler: 2 or 3 operands (1 or 2 ops)
        elif self.difficulty < 0.2:
            num_operands = random.randint(2, 4) # 2 to 4 operands (1 to 3 ops)
        else:
            num_operands = random.randint(3, 5) # 3 to 5 operands (2 to 4 ops)

        operands = [random.randint(1, 20) for _ in range(num_operands)]
        ops_choices = ['+', '-', '*', '/']
        chosen_ops = [random.choice(ops_choices) for _ in range(num_operands - 1)]

        expression_parts = [str(operands[0])]
        for i in range(num_operands - 1):
            op = chosen_ops[i]
            operand2 = operands[i+1]
            # Prevent division by zero in the expression string itself
            if op == '/' and operand2 == 0:
                operand2 = 1 # Change 0 to 1 if it's a denominator
                operands[i+1] = 1 # Also update the list for potential later use if we were building value iteratively
            expression_parts.extend([op, str(operand2)])
        
        expression_str = " ".join(expression_parts)

        # Add parentheses randomly for expressions with 3 or more operands (2+ operations)
        # This makes precedence explicit and varies task structure.
        if num_operands >= 3 and random.random() < 0.5: # 50% chance to add parentheses
            # Simple parenthesizing strategies for now
            if num_operands == 3: # e.g., (a op b) op c OR a op (b op c)
                if random.random() < 0.5:
                    expression_str = f"({operands[0]} {chosen_ops[0]} {operands[1]}) {chosen_ops[1]} {operands[2]}"
                else:
                    expression_str = f"{operands[0]} {chosen_ops[0]} ({operands[1]} {chosen_ops[1]} {operands[2]})"
            elif num_operands == 4: # e.g., ((a op b) op c) op d OR (a op (b op c)) op d etc.
                # Could be more sophisticated, for now, just one set of parens
                idx_to_paren = random.randint(0,1) # 0 for first pair, 1 for second pair
                if idx_to_paren == 0:
                    expression_str = f"(({operands[0]} {chosen_ops[0]} {operands[1]}) {chosen_ops[1]} {operands[2]}) {chosen_ops[2]} {operands[3]}"
                else: # Group last two operations
                    expression_str = f"{operands[0]} {chosen_ops[0]} ({operands[1]} {chosen_ops[1]} {operands[2]} {chosen_ops[2]} {operands[3]})"
            # For num_operands == 5, could add more complex parenthesizing, or nested.
            # Keeping it simpler for now.

        # --- CRITICAL FIX: Calculate expected_answer by evaluating expression_str ---
        try:
            # Replace / with // for eval if we want integer-like division behavior for whole numbers,
            # but problem statement implies float results, so direct eval is fine.
            # Ensure the expression is safe for eval. Since we construct it from numbers and basic ops, it should be.
            # Handle potential true division leading to floats.
            
            # Create a 'safe' eval environment
            safe_dict = {
                "__builtins__": None # No builtins allowed
            }
            # Allow basic math functions if needed, but for now, only operators
            
            # Ensure division by zero in the string to be eval'd is handled if it slips through
            # (though the operand2 check above should prevent it in the string itself)
            # A more robust way is to parse and compute, but eval is used for simplicity here.
            if "/ 0" in expression_str and not "/ 0.0" in expression_str : # Avoid division by zero literal in string
                 return self._generate_arithmetic() # Retry generation

            evaluated_value = eval(expression_str, safe_dict, {}) # Evaluate in a restricted scope
            expected_answer = round(float(evaluated_value), 4)
        except ZeroDivisionError:
            # print(f"Warning: ZeroDivisionError during eval for expression: {expression_str}. Retrying generation.")
            return self._generate_arithmetic() # Retry if eval causes division by zero
        except Exception as e:
            # print(f"Error evaluating expression '{expression_str}': {e}. Retrying generation.")
            # traceback.print_exc() # For debugging
            return self._generate_arithmetic() # Retry on any eval error

        task = f"Calculate: {expression_str}"
        return task, expected_answer

    def _generate_algebra(self) -> Tuple[str, float]:
        # (Expanded algebra forms from previous good version)
        problem_type = random.choice(['ax+b=c', 'ax-b=c', 'b-ax=c', '(ax+b)/d=c', 'a(x+b)=c', 'x/a+b=c', 'x/a-b=c'])
        
        x_val = float(random.randint(-10, 10))
        a = random.randint(1, 10) * random.choice([-1, 1]) 
        b = random.randint(-20, 20)
        expected_answer = x_val # Default, will be overridden if calculation changes it
        
        if problem_type == 'ax+b=c':
            c = a * x_val + b
            task = f"Solve for x: {a}x + {b} = {c}"
        elif problem_type == 'ax-b=c':
            c = a * x_val - b
            task = f"Solve for x: {a}x - {b} = {c}"
        elif problem_type == 'b-ax=c':
            c = b - a * x_val
            task = f"Solve for x: {b} - {a}x = {c}"
        elif problem_type == '(ax+b)/d=c':
            d = random.randint(2, 5) * random.choice([-1, 1])
            while d == 0: d = random.randint(2, 5) * random.choice([-1, 1]) # Ensure d is not zero
            c_val_intermediate = (a * x_val + b) / d
            c = round(c_val_intermediate) 
            task = f"Solve for x: ({a}x + {b}) / {d} = {c}"
            if a == 0: return self._generate_algebra()
            expected_answer = (c * d - b) / a
        elif problem_type == 'a(x+b)=c':
            c_val_intermediate = a * (x_val + b)
            c = round(c_val_intermediate)
            task = f"Solve for x: {a}(x + {b}) = {c}"
            if a == 0: return self._generate_algebra()
            expected_answer = (c / a) - b
        elif problem_type == 'x/a+b=c':
            if a == 0: a = random.randint(1,5) * random.choice([-1,1]) # ensure a is not zero
            c = (x_val / a) + b
            c = round(c, 2) # Round c for cleaner problem statement
            task = f"Solve for x: x / {a} + {b} = {c}"
            expected_answer = (c - b) * a
        elif problem_type == 'x/a-b=c':
            if a == 0: a = random.randint(1,5) * random.choice([-1,1]) # ensure a is not zero
            c = (x_val / a) - b
            c = round(c, 2)
            task = f"Solve for x: x / {a} - {b} = {c}"
            expected_answer = (c + b) * a


        expected_answer = round(float(expected_answer), 4)
        return task, expected_answer
    
    def _generate_word_problem(self) -> Tuple[str, float]:
        # (Expanded word problems from previous good version)
        templates = [
            ("A train travels at {speed} miles per hour. How far will it travel in {time} hours?", lambda s, t: s * t),
            ("If you have ${amount} and spend ${spent}, how much money do you have left?", lambda a, s: a - s),
            ("A rectangle has a length of {length} units and a width of {width} units. What is its area?", lambda l, w: l * w),
            ("If a book costs ${cost} and there is a {discount}% discount, what is the final price?", lambda c, d: c * (1 - d/100.0)),
            ("A recipe calls for {flour_cups} cups of flour for {servings} servings. How much flour is needed per serving?", lambda fc, s: fc / s if s != 0 else 0),
            ("You buy {items} items, each costing ${price:.2f}. What is the total cost?", lambda i, p: i * p), # Added .2f for price
            ("A car travels {distance} km on {liters} liters of fuel. What is its fuel efficiency in km per liter?", lambda d, l: d / l if l != 0 else 0),
            ("A worker earns ${rate:.2f} per hour and works {hours} hours. What are their total earnings?", lambda r, h: r * h)
        ]
        template, formula = random.choice(templates)
        params = {}
        if "speed" in template: params["speed"] = random.randint(30, 100)
        if "time" in template: params["time"] = round(random.uniform(0.5, 5.0), 1)
        if "amount" in template: params["amount"] = random.randint(50, 200)
        if "spent" in template: params["spent"] = random.randint(10, params.get("amount", 50) -1 ) 
        if "length" in template: params["length"] = random.randint(5, 20)
        if "width" in template: params["width"] = random.randint(3, 15)
        if "cost" in template: params["cost"] = random.randint(20, 200)
        if "discount" in template: params["discount"] = random.randint(5, 50)
        if "flour_cups" in template: params["flour_cups"] = random.randint(1, 8)
        if "servings" in template: params["servings"] = random.randint(1, 12)
        if "items" in template: params["items"] = random.randint(2,10)
        if "price" in template: params["price"] = round(random.uniform(1.0, 20.0), 2)
        if "distance" in template: params["distance"] = random.randint(50, 500)
        if "liters" in template: params["liters"] = random.randint(5, 50)
        if "rate" in template: params["rate"] = round(random.uniform(10.0, 50.0), 2)
        if "hours" in template: params["hours"] = round(random.uniform(1.0, 40.0), 1)
        
        task = template.format(**params)
        
        func_args = []
        if "speed" in template and "time" in template: func_args = [params["speed"], params["time"]]
        elif "amount" in template and "spent" in template: func_args = [params["amount"], params["spent"]]
        elif "length" in template and "width" in template: func_args = [params["length"], params["width"]]
        elif "cost" in template and "discount" in template: func_args = [params["cost"], params["discount"]]
        elif "flour_cups" in template and "servings" in template: func_args = [params["flour_cups"], params["servings"]]
        elif "items" in template and "price" in template: func_args = [params["items"], params["price"]]
        elif "distance" in template and "liters" in template: func_args = [params["distance"], params["liters"]]
        elif "rate" in template and "hours" in template: func_args = [params["rate"], params["hours"]]

        expected_answer = round(formula(*func_args), 4) if func_args else 0.0
        return task, expected_answer
    
    def verify(self, task: str, answer: str, expected_math_answer: Any) -> Tuple[bool, float, float]:
        # (Verification logic remains the same as your provided version)
        syntax_reward = 0.0 
        is_correct = False
        confidence = 0.0

        try:
            user_answer_str = answer.strip()
            match = re.search(r'-?\d+\.?\d*', user_answer_str)
            if match:
                user_answer_val = float(match.group(0))
                syntax_reward = 0.05 
            else: 
                return False, 0.0, 0.0 

            if isinstance(expected_math_answer, (int, float)):
                # Use a slightly more generous tolerance for floating point comparisons
                tolerance = max(0.01, abs(expected_math_answer * 0.05)) # 5% tolerance or 0.01 absolute
                is_correct = abs(user_answer_val - expected_math_answer) <= tolerance
                if is_correct:
                    confidence = 1.0
                    syntax_reward = 0.1 
                else:
                    diff_ratio = abs(user_answer_val - expected_math_answer) / max(1.0, abs(expected_math_answer)) if expected_math_answer != 0 else abs(user_answer_val)
                    confidence = max(0.0, 1.0 - min(1.0, diff_ratio * 2)) 
            else: 
                # print(f"Warning: expected_math_answer is not a number: {expected_math_answer}")
                return False, 0.0, syntax_reward 
            return is_correct, confidence, syntax_reward
        except Exception:
            # print(f"Error in math verification: {e} for answer '{answer}'")
            return False, 0.0, 0.0


class CodeTaskTemplate(TaskTemplate):
    """Template for generating coding tasks."""
    
    def __init__(self, difficulty: float = 0.5):
        super().__init__(difficulty)
        
    def generate(self) -> Tuple[str, Any]: 
        if self.difficulty < 0.4: 
            return self._generate_simple_function()
        elif self.difficulty < 0.7:
            return self._generate_algorithm()
        else:
            return self._generate_data_structure()
    
    def _generate_simple_function(self) -> Tuple[str, Dict]:
        # (Using the expanded list from your latest good version)
        function_templates = [
            {
                "name": "add_two_numbers",
                "description": "Write a Python function named 'add_two_numbers' that takes two numbers and returns their sum.",
                "signature": "def add_two_numbers(a, b):", 
                "test_cases": [{"input": (3, 5), "output": 8}, {"input": (-1, 1), "output": 0}, {"input": (0.5, 0.5), "output": 1.0}],
                "solution": "def add_two_numbers(a, b):\n    return a + b"
            },
            {
                "name": "get_string_length",
                "description": "Write a Python function named 'get_string_length' that takes a string and returns its length.",
                "signature": "def get_string_length(s: str) -> int:",
                "test_cases": [{"input": ("hello",), "output": 5}, {"input": ("",), "output": 0}, {"input": ("BitZero",), "output": 7}],
                "solution": "def get_string_length(s: str) -> int:\n    return len(s)"
            },
            {
                "name": "is_positive",
                "description": "Write a Python function named 'is_positive' that returns True if a number is greater than 0, False otherwise.",
                "signature": "def is_positive(num) -> bool:",
                "test_cases": [{"input": (5,), "output": True}, {"input": (-5,), "output": False}, {"input": (0,), "output": False}],
                "solution": "def is_positive(num) -> bool:\n    return num > 0"
            },
            {
                "name": "multiply_by_two",
                "description": "Write a Python function named 'multiply_by_two' that takes a number and returns the number multiplied by 2.",
                "signature": "def multiply_by_two(num):",
                "test_cases": [{"input": (4,), "output": 8}, {"input": (-3,), "output": -6}, {"input": (0,), "output": 0}],
                "solution": "def multiply_by_two(num):\n    return num * 2"
            },
            {
                "name": "get_first_element",
                "description": "Write a Python function named 'get_first_element' that returns the first element of a list. Return None if the list is empty.",
                "signature": "def get_first_element(data: list):",
                "test_cases": [{"input": ([1, 2, 3],), "output": 1}, {"input": (["a", "b"],), "output": "a"}, {"input": ([],), "output": None}],
                "solution": "def get_first_element(data: list):\n    if data:\n        return data[0]\n    return None"
            },
            {
                "name": "concatenate_strings",
                "description": "Write a Python function named 'concatenate_strings' that takes two strings and returns them concatenated.",
                "signature": "def concatenate_strings(s1: str, s2: str) -> str:",
                "test_cases": [{"input": ("hello", " world"), "output": "hello world"}, {"input": ("Bit", "Zero"), "output": "BitZero"}],
                "solution": "def concatenate_strings(s1: str, s2: str) -> str:\n    return s1 + s2"
            },
            {
                "name": "check_if_even",
                "description": "Write a Python function named 'check_if_even' that returns True if a number is even, False otherwise.",
                "signature": "def check_if_even(num: int) -> bool:",
                "test_cases": [{"input": (2,), "output": True}, {"input": (3,), "output": False}, {"input": (0,), "output": True}],
                "solution": "def check_if_even(num: int) -> bool:\n    return num % 2 == 0"
            },
            {
                "name": "find_max_of_two",
                "description": "Write a Python function named 'find_max_of_two' that returns the larger of two numbers.",
                "signature": "def find_max_of_two(a, b):",
                "test_cases": [{"input": (3, 7), "output": 7}, {"input": (10, 5), "output": 10}, {"input": (-1, -5), "output": -1}],
                "solution": "def find_max_of_two(a, b):\n    if a > b:\n        return a\n    return b"
            },
            {
                "name": "greet_user",
                "description": "Write a Python function named 'greet_user' that takes a name (string) and returns 'Hello, [name]!'.",
                "signature": "def greet_user(name: str) -> str:",
                "test_cases": [{"input": ("Alice",), "output": "Hello, Alice!"}, {"input": ("BitZero",), "output": "Hello, BitZero!"}],
                "solution": "def greet_user(name: str) -> str:\n    return f\"Hello, {name}!\"" 
            },
            {
                "name": "calculate_square",
                "description": "Write a Python function named 'calculate_square' that returns the square of a number.",
                "signature": "def calculate_square(num):",
                "test_cases": [{"input": (3,), "output": 9}, {"input": (-4,), "output": 16}, {"input": (0,), "output": 0}],
                "solution": "def calculate_square(num):\n    return num * num"
            },
            {
                "name": "is_string_empty",
                "description": "Write a Python function named 'is_string_empty' that checks if a string is empty.",
                "signature": "def is_string_empty(s: str) -> bool:",
                "test_cases": [{"input": ("",), "output": True}, {"input": (" ",), "output": False}, {"input": ("text",), "output": False}],
                "solution": "def is_string_empty(s: str) -> bool:\n    return len(s) == 0"
            },
             {
                "name": "sum_of_list", 
                "description": "Write a Python function named 'sum_of_list' that returns the sum of all numerical elements in a list. The function should handle empty lists by returning 0.",
                "signature": "def sum_of_list(numbers: list) -> float:",
                "test_cases": [{"input": ([1, 2, 3, 4, 5],), "output": 15.0}, {"input": ([],), "output": 0.0}],
                "solution": "def sum_of_list(numbers: list) -> float:\n    return float(sum(numbers))"
            },
            {
                "name": "is_palindrome_simple", 
                "description": "Write a Python function named 'is_palindrome_simple' that checks if a given string is a palindrome (reads the same forwards and backwards). Consider only alphanumeric characters and ignore case.",
                "signature": "def is_palindrome_simple(text: str) -> bool:",
                "test_cases": [{"input": ("racecar",), "output": True}, {"input": ("A man, a plan, a canal: Panama",), "output": True}],
                "solution": "def is_palindrome_simple(text: str) -> bool:\n    s = ''.join(filter(str.isalnum, text)).lower()\n    return s == s[::-1]"
            }
        ]
        template = random.choice(function_templates)
        task = f"{template['description']}\nComplete the following function signature:\n{template['signature']}\n    # Your code here"
        return task, template

    def _generate_algorithm(self) -> Tuple[str, Dict]:
        # (Using the expanded algorithm templates from previous good version)
        algorithm_templates = [
            {
                "name": "binary_search",
                "description": "Implement a Python function named 'binary_search' that finds the index of a target value in a sorted list of integers. Return -1 if the target is not found.",
                "signature": "def binary_search(arr: list, target: int) -> int:",
                "test_cases": [
                    {"input": ([1, 2, 3, 4, 5], 3), "output": 2},
                    {"input": ([1, 2, 3, 4, 5], 6), "output": -1},
                    {"input": ([], 5), "output": -1},
                    {"input": ([5], 5), "output": 0},
                    {"input": ([2, 5, 7, 8, 11, 12], 13), "output": -1},
                    {"input": ([2, 5, 7, 8, 11, 12], 7), "output": 2}
                ],
                "solution": "def binary_search(arr: list, target: int) -> int:\n    left, right = 0, len(arr) - 1\n    while left <= right:\n        mid = (left + right) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            left = mid + 1\n        else:\n            right = mid - 1\n    return -1"
            },
            {
                "name": "factorial_iterative",
                "description": "Implement an iterative Python function named 'factorial_iterative' that calculates the factorial of a non-negative integer n (n!).",
                "signature": "def factorial_iterative(n: int) -> int:",
                "test_cases": [
                    {"input": (0,), "output": 1}, {"input": (1,), "output": 1},
                    {"input": (5,), "output": 120}, {"input": (7,), "output": 5040}
                ],
                "solution": "def factorial_iterative(n: int) -> int:\n    if n < 0:\n        raise ValueError(\"Factorial not defined for negative numbers\")\n    if n == 0:\n        return 1\n    result = 1\n    for i in range(1, n + 1):\n        result *= i\n    return result"
            },
            {
                "name": "bubble_sort",
                "description": "Implement the Bubble Sort algorithm in Python to sort a list of numbers in ascending order.",
                "signature": "def bubble_sort(arr: list) -> list:",
                "test_cases": [
                    {"input": ([5, 1, 4, 2, 8],), "output": [1, 2, 4, 5, 8]},
                    {"input": ([],), "output": []},
                    {"input": ([1, 2, 3],), "output": [1, 2, 3]},
                    {"input": ([9,0,3,7,2,18,11],), "output": [0,2,3,7,9,11,18]}
                ],
                "solution": "def bubble_sort(arr: list) -> list:\n    n = len(arr)\n    for i in range(n):\n        swapped = False\n        for j in range(0, n-i-1):\n            if arr[j] > arr[j+1]:\n                arr[j], arr[j+1] = arr[j+1], arr[j]\n                swapped = True\n        if not swapped:\n            break\n    return arr"
            }
        ]
        template = random.choice(algorithm_templates)
        task = f"{template['description']}\nComplete the following function signature:\n{template['signature']}\n    # Your code here"
        return task, template

    def _generate_data_structure(self) -> Tuple[str, Dict]:
        # (Using the Stack example, can be expanded)
        data_structure_templates = [
            {
                "name": "Stack", 
                "description": "Implement a Python Stack class named 'Stack'. It requires the following methods: __init__ (no arguments); push(self, item); pop(self) which returns the item or raises IndexError if empty; peek(self) which returns the top item or raises IndexError if empty; and is_empty(self) which returns a boolean. Use a list internally for storage.",
                "signature": "class Stack:\\n    # Your code here to define the class and its methods", # Much shorter
                "test_code": 
                    "s = Stack()\\n"
                    "assert s.is_empty() == True\\n"
                    "s.push(1)\\n"
                    "assert s.is_empty() == False\\n"
                    "s.push('two')\\n"
                    "assert s.peek() == 'two'\\n"
                    "assert s.pop() == 'two'\\n"
                    "assert s.pop() == 1\\n"
                    "assert s.is_empty() == True\\n"
                    "try:\\n    s.pop()\\n    assert False, 'Popping from empty stack should raise IndexError'\\nexcept IndexError:\\n    assert True\\n"
                    "try:\\n    s.peek()\\n    assert False, 'Peeking empty stack should raise IndexError'\\nexcept IndexError:\\n    assert True\\n",
                "solution": "class Stack:\\n    def __init__(self):\\n        self._items = []\\n\\n    def push(self, item: Any):\\n        self._items.append(item)\\n\\n    def pop(self) -> Any:\\n        if not self.is_empty():\\n            return self._items.pop()\\n        else:\\n            raise IndexError('pop from empty stack')\\n\\n    def peek(self) -> Any:\\n        if not self.is_empty():\\n            return self._items[-1]\\n        else:\\n            raise IndexError('peek from empty stack')\\n\\n    def is_empty(self) -> bool:\\n        return len(self._items) == 0"
            },
        ]
        template = random.choice(data_structure_templates)
        task = f"{template['description']}\nComplete the following class structure:\n{template['signature']}"
        return task, template
    
    def verify(self, task_description: str, answer_code: str, expected_template_data: Dict) -> Tuple[bool, float, float]:
        # (Verification logic remains the same as your provided version)
        is_functionally_correct = False
        confidence = 0.0
        syntax_reward = 0.0

        try:
            ast.parse(answer_code)
            syntax_reward = 0.15 
        except SyntaxError:
            return False, 0.0, 0.0 

        try:
            local_env = {}
            exec(answer_code, {}, local_env) # Execute the user's code

            if "test_cases" in expected_template_data: # Function verification
                function_name = expected_template_data["name"]
                test_cases = expected_template_data["test_cases"]

                if function_name not in local_env:
                    return False, 0.0, syntax_reward 
                if not test_cases: # Should not happen if templates are well-defined
                     return False, 0.5, syntax_reward # Can't verify, but syntax was ok

                passed_tests = 0
                for test_case in test_cases:
                    test_input = test_case["input"]
                    expected_output = test_case["output"]
                    try:
                        if isinstance(test_input, tuple):
                            actual_output = local_env[function_name](*test_input)
                        else: 
                            actual_output = local_env[function_name](test_input)
                        
                        if actual_output == expected_output:
                            passed_tests += 1
                    except Exception: # Error during test case execution
                        continue # Count as failed test
                
                if test_cases: 
                    confidence = passed_tests / len(test_cases)
                
                if confidence >= 0.999: # All tests passed (allow for float issues if any)
                    is_functionally_correct = True
                    syntax_reward = 0.25 
                elif confidence > 0: # Some tests passed
                    is_functionally_correct = False 
                    syntax_reward = 0.20 
            
            elif "test_code" in expected_template_data: # Class verification
                class_name = expected_template_data["name"] # Assumes class name is in 'name'
                test_code_to_run = expected_template_data["test_code"]

                if class_name not in local_env: 
                    return False, 0.0, syntax_reward

                try:
                    exec(test_code_to_run, local_env) 
                    is_functionally_correct = True
                    confidence = 1.0
                    syntax_reward = 0.25 
                except AssertionError:
                    confidence = 0.3 
                    is_functionally_correct = False
                except Exception:
                    confidence = 0.1 
                    is_functionally_correct = False
            
            else: 
                return False, 0.0, syntax_reward 
        except Exception: 
            return False, 0.0, syntax_reward 

        return is_functionally_correct, confidence, syntax_reward


class TaskGenerator:
    """Task generator for BitZero self-play."""
    
    def __init__(self, 
                 initial_difficulty: float = 0.1,
                 max_difficulty: float = 0.9,
                 difficulty_step: float = 0.05):
        self.current_difficulty = initial_difficulty
        self.max_difficulty = max_difficulty
        self.difficulty_step = difficulty_step
        self.task_templates = {
            "math": MathTaskTemplate(difficulty=self.current_difficulty),
            "code": CodeTaskTemplate(difficulty=self.current_difficulty)
        }
        self.task_history = [] 
        
    def generate_task(self, task_type: Optional[str] = None) -> Tuple[str, Any, str]:
        if task_type is None:
            task_type = random.choice(list(self.task_templates.keys()))
        
        self.task_templates[task_type].difficulty = self.current_difficulty
        task_description, expected_answer_data = self.task_templates[task_type].generate()
        
        self.task_history.append({
            "type": task_type, "task": task_description, 
            "expected_answer_data": expected_answer_data, 
            "difficulty": self.current_difficulty
        })
        return task_description, expected_answer_data, task_type
    
    def adjust_difficulty(self, success_metric: float): 
        # Current logic: if success_metric (e.g. avg_loss) is high (>0.75), DECREASE difficulty
        # If success_metric is low (<0.25), INCREASE difficulty
        # This seems reversed if success_metric is loss. Let's assume success_metric is reward-like (higher is better)
        # For now, let's use the existing logic but be mindful of what success_metric represents.
        # If success_metric is loss, then:
        # if loss < 0.5 (good performance): increase difficulty
        # if loss > 2.0 (bad performance): decrease difficulty
        # The provided training script uses loss, so let's adjust based on that.
        # The current trainer calls adjust_difficulty with 0.8 if loss < 0.5 (good)
        # and 0.2 if loss > 2.0 (bad).

        if success_metric > 0.75: # Corresponds to low loss / good performance
            self.current_difficulty = min(self.max_difficulty, self.current_difficulty + self.difficulty_step)
        elif success_metric < 0.25: # Corresponds to high loss / bad performance
            self.current_difficulty = max(0.05, self.current_difficulty - self.difficulty_step) 
        
        for template in self.task_templates.values():
            template.difficulty = self.current_difficulty


class TaskVerifier:
    """Task verification system for BitZero self-play."""
    
    def __init__(self):
        self.math_task_template = MathTaskTemplate() 
        self.code_task_template = CodeTaskTemplate() 

    def verify(self, task_description: str, answer_str: str, expected_answer_data: Any, task_type: str) -> Tuple[bool, float]:
        # (Using the combined reward logic from your previous good version)
        is_fully_correct = False
        final_reward = 0.0
        confidence = 0.0 
        syntax_or_parse_reward = 0.0 

        if task_type == "math":
            is_fully_correct, confidence, syntax_or_parse_reward = self.math_task_template.verify(task_description, answer_str, expected_answer_data)
            if syntax_or_parse_reward > 0: 
                final_reward = syntax_or_parse_reward + (confidence * 0.9) 
            else:
                final_reward = 0.0 
            
        elif task_type == "code":
            is_fully_correct, confidence, syntax_or_parse_reward = self.code_task_template.verify(task_description, answer_str, expected_answer_data)
            final_reward = syntax_or_parse_reward 
            if syntax_or_parse_reward > 0: 
                final_reward += (confidence * 0.75) 
        else:
            # print(f"Unknown task type for verification: {task_type}")
            return False, 0.0

        final_reward = max(0.0, min(1.0, final_reward)) 
        
        # Optional: print detailed verification breakdown
        # print(f"Verify - Task: {task_type}, Correct: {is_fully_correct}, Reward: {final_reward:.2f}, Conf: {confidence:.2f}, Syntax: {syntax_or_parse_reward:.2f}")
        return is_fully_correct, final_reward


if __name__ == "__main__":
    # Test the updated task generator
    generator = TaskGenerator(initial_difficulty=0.1) 
    verifier = TaskVerifier()
    
    print("--- Testing Math Task (Corrected Targets & Variety) ---")
    for i in range(10): # Generate more math tasks
        # Vary difficulty to test different generation paths
        if i < 3:
            generator.current_difficulty = 0.1 # Test low difficulty arithmetic
        elif i < 7:
            generator.current_difficulty = 0.4 # Test medium difficulty algebra
        else:
            generator.current_difficulty = 0.8 # Test higher difficulty word problems
            
        math_task_desc, math_expected_val, math_type = generator.generate_task("math")
        print(f"Math Task ({i+1}, Diff: {generator.current_difficulty:.2f}): {math_task_desc}")
        print(f"Expected Numerical Answer: {math_expected_val}")
        
        # Test verification with the correct answer
        correct, reward = verifier.verify(math_task_desc, str(math_expected_val), math_expected_val, math_type)
        print(f"Verification (with correct answer): Correct: {correct}, Reward: {reward:.4f}")
        
        # Test verification with a slightly off answer
        slightly_off_answer = str(math_expected_val + 0.5 + random.random())
        correct_off, reward_off = verifier.verify(math_task_desc, slightly_off_answer, math_expected_val, math_type)
        print(f"Verification (with '{slightly_off_answer}'): Correct: {correct_off}, Reward: {reward_off:.4f}\n")


    print("\n--- Testing Code Task (Expanded Simple Functions) ---")
    for i in range(10): 
        generator.current_difficulty = random.uniform(0.05, 0.35) 
        code_task_desc, code_expected_template, code_type = generator.generate_task("code")
        
        if "test_cases" not in code_expected_template and generator.current_difficulty < 0.4:
             print(f"Warning: Expected simple function for code task at diff {generator.current_difficulty:.2f}, but got other type. Retrying once.")
             code_task_desc, code_expected_template, code_type = generator.generate_task("code")
             if "test_cases" not in code_expected_template:
                 print("Still not a simple function after retry, skipping this iteration.")
                 continue

        print(f"Code Task ({i+1}, Diff: {generator.current_difficulty:.2f}): {code_task_desc}")
        correct_solution = code_expected_template.get("solution", "def placeholder(): pass")
        
        correct, reward = verifier.verify(code_task_desc, correct_solution, code_expected_template, code_type)
        print(f"Verification (with correct solution):\n{correct_solution[:100]}...\n-> Correct: {correct}, Reward: {reward:.4f}\n")
