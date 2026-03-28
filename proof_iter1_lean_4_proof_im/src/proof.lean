import Mathlib.Tactic
import Mathlib.Algebra.BigOperators.Group.Finset
import Mathlib.Algebra.BigOperators.Intervals
import Mathlib.Analysis.Convex.Jensen
import Mathlib.Analysis.Convex.SpecificFunctions.Basic

open Finset BigOperators

/-! # IMB Decomposition and Convex Cost Optimality

Formal verification of two foundational theorems for the
Parametric Convex-Cost Load Smoothing hypothesis:

1. **Gauss Sum / IMB Decomposition**: Total IMB decomposes via
   the arithmetic series formula into a function of dependency distances.

2. **Jensen-based Convex Cost Optimality**: Under any strictly convex
   cost function (α > 1), uniform load distribution minimizes total cost.
-/

/-! ## Part 1: Gauss Sum and IMB Decomposition -/

/-- Gauss sum (multiplication form): (∑_{i=0}^{d} i) * 2 = d * (d + 1).
    Uses multiplication to avoid natural number division issues. -/
lemma gauss_sum_mul_two (d : ℕ) :
    (∑ i in Finset.range (d + 1), i) * 2 = d * (d + 1) := by
  have h := Finset.sum_range_id_mul_two (d + 1)
  simp only [Nat.add_sub_cancel] at h
  linarith

/-- IMB decomposition: for dependencies with distances d_1,...,d_m,
    2 * ∑_k ∑_{i=0}^{d_k} i = ∑_k d_k * (d_k + 1).
    Total IMB decomposes into per-dependency Gauss contributions. -/
theorem imb_decomposition {m : ℕ} (dist : Fin m → ℕ) :
    (∑ k : Fin m, ∑ i in Finset.range (dist k + 1), i) * 2 =
    ∑ k : Fin m, dist k * (dist k + 1) := by
  rw [Finset.sum_mul]
  congr 1; ext k; exact gauss_sum_mul_two (dist k)

/-- Summation order equivalence: swapping nested finite sums.
    Position-centric IMB equals dependency-centric IMB. -/
theorem sum_order_swap {α : Type*} [AddCommMonoid α]
    {I J : Type*} (s : Finset I) (t : Finset J) (f : I → J → α) :
    ∑ i in s, ∑ j in t, f i j = ∑ j in t, ∑ i in s, f i j :=
  Finset.sum_comm

/-! ## Part 2: Convex Cost Optimality -/

/-- Power functions x^p are strictly convex on [0,∞) for p > 1.
    Direct instantiation of Mathlib's strictConvexOn_rpow. -/
lemma rpow_strict_convex_on {p : ℝ} (hp : 1 < p) :
    StrictConvexOn ℝ (Set.Ici (0 : ℝ)) (fun x : ℝ => x ^ p) :=
  strictConvexOn_rpow hp

/-- Jensen's inequality application: uniform load minimizes convex cost.
    For convex f on [0,∞) and non-negative x_1,...,x_n summing to S:
    n * f(S/n) ≤ ∑ f(x_i). -/
theorem jensen_uniform_optimal
    {n : ℕ} (hn : 0 < n)
    {f : ℝ → ℝ} (hf : ConvexOn ℝ (Set.Ici (0 : ℝ)) f)
    (x : Fin n → ℝ) (hx : ∀ i, 0 ≤ x i)
    (S : ℝ) (hS : ∑ i : Fin n, x i = S) :
    (n : ℝ) * f (S / (n : ℝ)) ≤ ∑ i : Fin n, f (x i) := by
  have hn_pos : (0 : ℝ) < (n : ℝ) := Nat.cast_pos.mpr hn
  have hn_ne : (n : ℝ) ≠ 0 := ne_of_gt hn_pos
  -- Weights non-negative
  have hw_nn : ∀ i ∈ (Finset.univ : Finset (Fin n)), 0 ≤ (n : ℝ)⁻¹ :=
    fun _ _ => le_of_lt (inv_pos.mpr hn_pos)
  -- Weights sum to 1
  have hw_sum : ∑ _i : Fin n, ((n : ℝ)⁻¹ : ℝ) = 1 := by
    rw [Finset.sum_const, Finset.card_fin, nsmul_eq_mul]
    exact mul_inv_cancel₀ hn_ne
  -- Points in convex set
  have hx_mem : ∀ i ∈ (Finset.univ : Finset (Fin n)), x i ∈ Set.Ici (0 : ℝ) :=
    fun i _ => Set.mem_Ici.mpr (hx i)
  -- Apply Jensen's inequality
  have hj := hf.map_sum_le hw_nn hw_sum hx_mem
  -- hj : f (∑ i, (n:ℝ)⁻¹ • x i) ≤ ∑ i, (n:ℝ)⁻¹ • f (x i)
  -- Simplify smul to mul and factor constants out of sums
  simp only [smul_eq_mul, ← Finset.mul_sum] at hj
  -- hj : f ((n:ℝ)⁻¹ * ∑ i, x i) ≤ (n:ℝ)⁻¹ * ∑ i, f (x i)
  rw [hS, ← div_eq_inv_mul] at hj
  -- hj : f (S / n) ≤ (n:ℝ)⁻¹ * ∑ i, f (x i)
  -- Multiply both sides by n
  calc (n : ℝ) * f (S / (n : ℝ))
      ≤ (n : ℝ) * ((n : ℝ)⁻¹ * ∑ i : Fin n, f (x i)) :=
        mul_le_mul_of_nonneg_left hj (le_of_lt hn_pos)
    _ = ∑ i : Fin n, f (x i) := by
        rw [← mul_assoc, mul_inv_cancel₀ hn_ne, one_mul]

/-- Linear cost negative control: at α=1, total cost = S
    regardless of how load is distributed. -/
theorem linear_cost_invariant
    {n : ℕ} (x : Fin n → ℝ) (S : ℝ) (hS : ∑ i : Fin n, x i = S) :
    ∑ i : Fin n, x i = S := hS
