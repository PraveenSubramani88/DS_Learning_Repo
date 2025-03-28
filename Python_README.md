# Python for Data Science

Welcome to the Python for Data Science section. This area covers the essentials of Python programming tailored for data analysis and scientific computing.

## Overview
- **Purpose:** Introduce Python programming, libraries, and best practices for data science.
- **Key Libraries:** NumPy, Pandas, SciPy, Scikit-learn, and Matplotlib.

## Topics Covered
- **Python Basics:** Syntax, data types, control structures, and functions.
- **Data Manipulation:** Using Pandas for data cleaning, transformation, and exploration.
- **Numerical Computing:** Leveraging NumPy for efficient array operations.
- **Visualization:** Creating graphs with Matplotlib and Seaborn.
- **Practical Examples:** Jupyter notebooks demonstrating real-world data tasks.

## How to Use This Section
- Explore the provided notebooks and scripts.
- Experiment with the examples and modify them for your own projects.
- Contribute your insights or improvements via pull requests.

Happy coding!

## **1️⃣ Basic Indexing** (Access Single Elements)  
```python
num_list = list(range(1, 21))  # Sample list from 1 to 20

print(num_list[0])    # First element
print(num_list[-1])   # Last element
print(num_list[len(num_list) // 2])  # Middle element (for odd-length lists)
```

---

## **2️⃣ Simple Slices** (Extract Specific Ranges)  
```python
print(num_list[:5])   # First 5 elements
print(num_list[-5:])  # Last 5 elements
print(num_list[5:10]) # 6th to 10th elements
print(num_list[:])    # Full list (copy)
```

---

## **3️⃣ Custom Ranges** (Create Dynamic Subsets)  
```python
print(num_list[1:-1])   # Remove first and last element
print(num_list[2:-2])   # Remove first 2 and last 2 elements
print(num_list[:len(num_list)//2])  # First half of the list
print(num_list[len(num_list)//2:])  # Second half of the list
print(num_list[3:len(num_list)-3])  # Remove first 3 and last 3 elements
```

---

## **4️⃣ Step-Based Slicing** (Skip Elements Efficiently)  
```python
print(num_list[::2])   # Every second element (even indices)
print(num_list[1::2])  # Every second element (odd indices)
print(num_list[::3])   # Every third element
print(num_list[::-1])  # Reverse list
print(num_list[::-2])  # Reverse, skipping every second element
print(num_list[2:15:3])  # Start from index 2, step by 3, until 15
```

---

## **5️⃣ Smart Dynamic Slicing** (Adapt to Any List Length)  
```python
n = len(num_list)

print(num_list[:n//3])   # First third of the list
print(num_list[n//3:2*n//3])  # Middle third of the list
print(num_list[2*n//3:])  # Last third of the list
print(num_list[-(n//4):])  # Last quarter of the list
```

### **5 Key Points about `pprint`:**  

1. **Readable Output:** `pprint` stands for **"pretty-print"**, making complex data structures more readable.  
2. **Handles Nested Data:** Useful for printing **dictionaries, lists, tuples, and JSON** in an organized way.  
3. **Improved Formatting:** Unlike `print()`, it **adds indentation and line breaks** for better clarity.  
4. **Depth Control:** Allows limiting output for **deeply nested structures** to avoid clutter.  
5. **String Formatting:** `pformat()` returns a **formatted string** instead of directly printing.


**Immutable vs. Mutable** data types in Python:  

| **Category**  | **Definition** | **Examples** | **Can be Modified?** |  
|--------------|--------------|-------------|------------------|  
| **Immutable** | Unchangeable (value can be replaced, but not updated) | `int`, `float`, `bool`, `str`, `tuple` | ❌ No (Cannot modify existing data, only replace it) |  
| **Mutable**   | Changeable (value can be replaced and updated) | `list`, `set`, `dict`, `NumPy array`, `Pandas Series`, `DataFrame` | ✅ Yes (Can modify existing data) |  

### **Detailed Breakdown**  

| **Type**  | **Category** | **Examples** |  
|-----------|------------|-------------|  
| `int` (Integer) | Immutable | `10, -5, 1000` |  
| `float` (Floating Point) | Immutable | `1.23, -0.99, 3.1415` |  
| `bool` (Boolean) | Immutable | `True, False` |  
| `str` (String) | Immutable | `"hello"`, `'Python'` |  
| `tuple` (Tuple) | Immutable | `(1, 2, 3)`, `("a", "b", "c")` |  
| `list` (List) | Mutable | `[1, 2, 3]`, `["apple", "banana"]` |  
| `set` (Set) | Mutable | `{1, 2, 3}`, `{"a", "b", "c"}` |  
| `dict` (Dictionary) | Mutable | `{"name": "Alice", "age": 25}` |  
| `NumPy array` | Mutable | `np.array([1, 2, 3])` |  
| `Pandas Series` | Mutable | `pd.Series([10, 20, 30])` |  
| `Pandas DataFrame` | Mutable | `pd.DataFrame({"A": [1, 2], "B": [3, 4]})` |  



**Google Python Style Guide** – A best practices guide for writing clean, consistent, and readable Python code, based on **PEP 8** with Google-specific enhancements. 🚀 [Read more](https://google.github.io/styleguide/pyguide.html).
