package lp

import (
	"errors"
	"fmt"
	"math"

	"github.com/gonum/floats"
	"github.com/gonum/matrix/mat64"
)

// TODO(btracey): Could have a solver structure with an abstract factorizer.
// This way could support sparse.
// Factorizer has: 1) Factorize, 2) SolveVec, 3) RankOneUpdate. If we had a rank-qne
// LQ update we could test both. Also need some way of extracting columns from a matrix
// Harder to define the phase-1 problem but must be possible.

//
// TODO(btracey): Have some sort of preprocessing step for helping to fix A to make it
// full rank?
// TODO(btracey): Reduce rows? Get rid of all zeros, places where only one variable
// is there, etc.
// TODO(btracey): Export this function.
// TODO(btracey): Need to improve error handling. Only want to panic if condition number inf.
// TODO(btracey): Instead of simplex solve, there should be a "Reduced ab" where
// the rows of a that are all zero are removed
// TODO(btracey): Provide method of artificial variables for help when problem
// is infeasible?
// TODO(btracey): Provide a "sanitize" function to do things like remove all
// zero rows and columns.
//
// Performance enhancements: Lots of linear solves.
// All involving linear solves of Ab. Should factor Ab once.
// Can make it not m^3?
// Sort of. Don't change ab that much, so can update ab factorization rather than
// Store a factorization of Ab instead of Ab itself. For example, the QR or LU
// decomposition. Then you just need to update the factorization at each step.
// The swap step is a rank-one update of Ab.
// Use rank-1 update to update QR factors.

var (
	ErrInfeasible = errors.New("lp: problem is infeasible")
	ErrLinSolve   = errors.New("lp: unexpected linear solve failure")
	ErrUnbounded  = errors.New("lp: problem is unbounded")
	ErrSingular   = errors.New("lp: A is singular")
	ErrZeroColumn = errors.New("lp: A has a column of all zeros")
	ErrZeroRow    = errors.New("lp: A has a row of all zeros")
)

var (
	badShape = "lp: size mismatch"
)

const (
	linDepTol = 1e-8

	initPosTol    = 1e-14 // tolerance on x being positive for the initial feasible. Needs to be >0 otherwise get singular.
	blandNegTol   = 1e-14
	unboundedTol  = 0
	moveNegTol    = 0
	moveZeroTol   = 0
	rRoundTol     = 1e-13 // round r to 0
	dRoundTol     = 1e-13
	phaseIZeroTol = 1e-14 // tolerance on x value of the phase I problem being zero
	blandZeroTol  = 1e-14 // tolerance on zero move
)

// simplex solves an LP in standard form:
//  minimize	c^T x
//  s.t. 		A*x = b
//  			x >= 0
// A must have full rank, and must not have any columns with all zeros.
//
// The Convert function can be used to transform an LP into standard form.
//
// initialBasic is a set of indices specifying an initial feasible solution.
// If supplied, the initial feasible solution must be feasible.
//
// For a detailed description of the Simplex method please see lectures 11-13 of
// UC Math 352 https://www.youtube.com/watch?v=ESzYPFkY3og&index=11&list=PLh464gFUoJWOmBYla3zbZbc4nv2AXez6X.
func simplex(initialBasic []int, c []float64, A mat64.Matrix, b []float64, tol float64) (float64, []float64, []int, error) {
	err := verifyInputs(initialBasic, c, A, b)
	if err != nil {
		if err == ErrUnbounded {
			return math.Inf(-1), nil, nil, ErrUnbounded
		}
		return math.NaN(), nil, nil, err
	}
	m, n := A.Dims()
	/*
		fmt.Printf("Ainit\n%0.4v\n", mat64.Formatted(A))
		fmt.Printf("a = %#v\n", A)
		fmt.Printf("b = %#v\n", b)
		fmt.Printf("c = %#v\n", c)
		fmt.Printf("initial basic = %#v\n", initialBasic)
		fmt.Println("tol = ", tol)
	*/

	// There is at least one optimal solution to the LP which is at the intersection
	// to a set of constraint boundaries. For a standard form LP with m variables
	// and n equality constraints, at least m-n elements of x must equal zero
	// at optimality. The Simplex algorithm solves the standard-form LP by starting
	// at an initial constraint vertex and successively moving to adjacent constraint
	// vertices. At every vertex, the set of non-zero x values are the "basic
	// feasible solution". The list of non-zero x's are maintained in basicIdxs,
	// the respective columns of A are in ab, and the actual non-zero values of
	// x are in xb.
	//
	// The LP is equality constrained such that A * x = b. This can be expanded
	// to
	//  ab * xb + an * xn = b
	// where ab are the columns of a in the basic set, and an are all of the
	// other columns. Since each element of xn is zero by definition, this means
	// that for all feasible solutions xb = ab^-1 * b.

	// Before the simplex algorithm can start, an initial feasible solution must
	// be found. If initialBasic is non-nil a feasible solution has been supplied.
	// If not, find an initial feasible solution, solving the "Phase I" problem
	// if necessary.
	var basicIdxs []int // The indices of the non-zero x values.
	var ab *mat64.Dense // The subset of columns of A listed in basicIdxs.
	var xb []float64    // The non-zero elements of x. xb = ab^-1 b
	if initialBasic != nil {
		if len(initialBasic) != m {
			panic("lp: incorrect number of initial vectors")
		}
		ab = extractColumns(A, initialBasic)
		xb, err = initializeFromBasic(ab, b)
		if err != nil {
			panic(err)
		}
		basicIdxs = make([]int, len(initialBasic))
		copy(basicIdxs, initialBasic)
	} else {
		basicIdxs, ab, xb, err = findInitialBasic(A, b)
		if err != nil {
			return math.NaN(), nil, nil, err
		}
	}
	fmt.Println("After initialize")
	fmt.Printf("Ainit\n%0.4v\n", mat64.Formatted(A))
	fmt.Printf("a = %#v\n", A)
	fmt.Printf("b = %#v\n", b)
	fmt.Printf("c = %#v\n", c)
	fmt.Printf("initial basic = %#v\n", basicIdxs)
	fmt.Println("tol = ", tol)

	// Now, basicIdxs contains the indexes for an initial feasible solution,
	// ab contains the extracted columns of A, and xb contains the feasible
	// solution (with all x ∉ basic equal to zero).

	// Construct some auxiliary variables. nonBasicIdx is the set of nonbasic
	// variables. cb is the subset of c for the basic variables. an and cn
	// are the equivalents to ab and cb but for the nonbasic variables.
	nonBasicIdx := make([]int, 0, n-m)
	inBasic := make(map[int]struct{})
	for _, v := range basicIdxs {
		inBasic[v] = struct{}{}
	}
	for i := 0; i < n; i++ {
		_, ok := inBasic[i]
		if !ok {
			nonBasicIdx = append(nonBasicIdx, i)
		}
	}
	// There is now an initial feasible set. First, construct some auxiliary variables.
	cb := make([]float64, len(basicIdxs))
	for i, idx := range basicIdxs {
		cb[i] = c[idx]
	}
	cn := make([]float64, len(nonBasicIdx))
	for i, idx := range nonBasicIdx {
		cn[i] = c[idx]
	}
	an := extractColumns(A, nonBasicIdx)

	bVec := mat64.NewVector(len(b), b)
	cbVec := mat64.NewVector(len(cb), cb)

	// Temporary data needed each iteration. (Descripbed later)
	r := make([]float64, n-m)
	move := make([]float64, m)
	lastCost := math.Inf(1)

	// Solve the linear program starting from the initial feasible set. This is
	// the "Phase 2" problem.
	//
	// Algorithm:
	// 1) Compute the "reduced costs" for the non-basic variables,
	// that is the lagrange multipliers of the constraints.
	// 		r = cn - an^T * ab^-T * cb
	// 2) If all of the reduced costs are positive, no improvement is possible,
	// and the solution is optimal (xn can only increase because of
	// non-negativity constraints).
	// 3) Choose the x_n with the most negative value of r. Call this value xe.
	// 4) xe will be increased from 0 until an element in xb will have to be 0.
	// This is choosing the constraint to move along. Each xb has some distance
	// xe can increase before it becomes negative. That distance can bound found
	// by
	//	xb = Ab^-1 b - Ab^-1 An xn
	//     = Ab^-1 b - Ab^-1 Ae xe
	//     = bhat + d x_e
	//  xe = bhat_i / - d_i
	// where Ae is the column of A corresponding to xe.
	// The constraining basic index is the first index for which this is true,
	// so remove the element which is min_i (bhat_i / -d_i), assuming d_i is negative.
	// If no d_i is less than 0, then the problem is unbounded.
	// 5) If the new xe is 0 (that is, bhat_i == 0), then this location is at
	// the intersection of several constraints. Use the Bland rule instead
	// of the rule in step 4 to avoid cycling.)

	/*
		fmt.Printf("ab pre =\n%0.4v\n", mat64.Formatted(ab))
		fmt.Printf("an pre =\n%0.4v\n", mat64.Formatted(an))
	*/
	for {
		// Compute reduced costs -- r = cn - an^T ab^-T cb
		var tmp mat64.Vector
		err := tmp.SolveVec(ab.T(), cbVec)
		if err != nil {
			panic("lp: unexpected linear solve error")
		}
		data := make([]float64, n-m)
		tmp2 := mat64.NewVector(n-m, data)
		tmp2.MulVec(an.T(), &tmp)
		floats.SubTo(r, cn, data)

		// Replace the most negative element in the simplex. If there are no
		// negative entries then the optimal solution has been found.
		minIdx := floats.MinIdx(r)
		if r[minIdx] >= -tol {
			break
		}

		// Round small numbers to 0
		for i, v := range r {
			if math.Abs(v) < rRoundTol {
				r[i] = 0
			}
		}

		/*
			fmt.Println("r = ", r)
			fmt.Println("b = ", b)
			fmt.Println("minidx = ", minIdx)
		*/

		// Compute the moving distance.
		err = computeMove(move, minIdx, A, ab, xb, nonBasicIdx)
		if err != nil {
			if err == ErrUnbounded {
				return math.Inf(-1), nil, nil, ErrUnbounded
			}
			panic(fmt.Sprintf("lp: unexpected error %s", err))
		}
		//fmt.Println("move = ", move)

		// Replace the basic index with the smallest move.
		replace := floats.MinIdx(move)
		fmt.Println("move = ", move)
		fmt.Println("r = ", r)
		if move[replace] <= moveZeroTol {
			// Can't move anywhere, need to use Bland rule instead, which is to
			// add in the smallest index with a negative r.
			// This is assuming that when the index is replaced, the resulting
			// ab is still singular. Instead, try each possible index in succession.
			// If the index can be replaced without creating a singular ab, do
			// so, otherwise try the next r.
			var found bool
			for i, v := range r {
				if v < -blandNegTol {
					minIdx = i
					found = true
					break
				}

			}
			fmt.Println("min idx = ", minIdx)
			if !found {
				panic("lp bland: no negative argument found")
			}
			err = computeMove(move, minIdx, A, ab, xb, nonBasicIdx)
			//fmt.Println("move = ", move)
			if err != nil {
				if err == ErrUnbounded {
					return math.Inf(-1), nil, nil, ErrUnbounded
				}
				panic(fmt.Sprintf("lp: unexpected error %s", err))
			}
			replace = floats.MinIdx(move)
			if math.Abs(move[replace]) < blandZeroTol {
				replace = -1
				// Find a zero index where replacement is non-singular.
				biCopy := make([]int, len(basicIdxs))
				for replaceTmp, v := range move {
					if v > blandZeroTol {
						continue
					}
					copy(biCopy, basicIdxs)
					biCopy[replaceTmp] = nonBasicIdx[minIdx]
					abTmp := extractColumns(A, biCopy)
					// If the condition number is reasonable, use this index.
					if mat64.Cond(abTmp, 1) < 1e16 {
						replace = replaceTmp
						break
					}
				}
				if replace == -1 {
					panic("lp: bland: all replacements cause ill-conditioned ab")
				}
			}
		}

		// Replace the constrained basicIdx with the newIdx.
		basicIdxs[replace], nonBasicIdx[minIdx] = nonBasicIdx[minIdx], basicIdxs[replace]
		cb[replace], cn[minIdx] = cn[minIdx], cb[replace]
		tmpCol1 := mat64.Col(nil, replace, ab)
		tmpCol2 := mat64.Col(nil, minIdx, an)
		ab.SetCol(replace, tmpCol2)
		an.SetCol(minIdx, tmpCol1)

		//fmt.Println("Ab = ")
		//fmt.Printf("% 0.4v\n", mat64.Formatted(ab))

		// Compute the new xb.
		xbVec := mat64.NewVector(len(xb), xb)
		err = xbVec.SolveVec(ab, bVec)
		if err != nil {
			panic("lp: unexpected linear solve error: " + err.Error())
		}

		// Verification check. Can remove when correct.
		cost := floats.Dot(cb, xb)
		if cost-lastCost > 1e-10 {
			panic("lp: cost should never increase")
		}
		lastCost = cost
	}
	// Found the optimum successfully. The basic variables get their values, and
	// the non-basic variables are all zero.
	opt := floats.Dot(cb, xb)
	xopt := make([]float64, n)
	for i, v := range basicIdxs {
		xopt[v] = xb[i]
	}
	return opt, xopt, basicIdxs, nil
}

func replaceBland() int {

}

// computeMove computes how far can be moved replacing the given index.
func computeMove(move []float64, minIdx int, A mat64.Matrix, ab *mat64.Dense, xb []float64, nonBasicIdx []int) error {
	// Find ae.
	col := mat64.Col(nil, nonBasicIdx[minIdx], A)
	aCol := mat64.NewVector(len(col), col)

	m, _ := ab.Dims()
	// d = - Ab^-1 Ae
	nb := m
	d := make([]float64, nb)
	dVec := mat64.NewVector(nb, d)
	err := dVec.SolveVec(ab, aCol)
	if err != nil {
		panic("lp: unexpected linear solve error")
	}
	floats.Scale(-1, d)

	for i, v := range d {
		if math.Abs(v) < dRoundTol {
			d[i] = 0
		}
	}

	// If no di < 0, then problem is unbounded.
	if floats.Min(d) >= unboundedTol {
		return ErrUnbounded
	}

	// move = bhat_i / - d_i, assuming d is negative.
	bHat := xb // ab^-1 b
	for i, v := range d {
		if v >= moveNegTol {
			move[i] = math.Inf(1)
		} else {
			move[i] = bHat[i] / math.Abs(v)
		}
	}
	return nil
}

func verifyInputs(initialBasic []int, c []float64, A mat64.Matrix, b []float64) error {
	// Verify inputs.
	m, n := A.Dims()
	if len(c) != n {
		panic("lp: c vector incorrect length")
	}
	if len(b) != m {
		panic("lp: b vector incorrect length")
	}
	if len(c) != n {
		panic("lp: c vector incorrect length")
	}
	if len(initialBasic) != 0 && len(initialBasic) != m {
		panic("lp: initialBasic incorrect length")
	}

	// Do some sanity checks so that ab does not become singular during the
	// simplex solution. If the ZeroRow checks are removed then the code for
	// finding a set of linearly indepent columns must be improved.

	// Check that if a row of A only has zero elements that corresponding
	// element in b is zero, otherwise the problem is infeasible.
	// Otherwise return ErrZeroRow.
	for i := 0; i < m; i++ {
		isZero := true
		for j := 0; j < n; j++ {
			if A.At(i, j) != 0 {
				isZero = false
				break
			}
		}
		if isZero && b[i] != 0 {
			// Infeasible
			return ErrInfeasible
		} else if isZero {
			return ErrZeroRow
		}
	}
	// Check that if a column only has zero elements that the respective C vector
	// is positive (otherwise unbounded). Otherwise return ErrZeroColumn.
	for j := 0; j < n; j++ {
		isZero := true
		for i := 0; i < m; i++ {
			if A.At(i, j) != 0 {
				isZero = false
				break
			}
		}
		if isZero && c[j] < 0 {
			return ErrUnbounded
		} else if isZero {
			return ErrZeroColumn
		}
	}
	return nil
}

// initializeFromBasic initializes the basic feasible solution given a set of
// basic indices. It extracts the columns
// of A specified by basicIdxs and finds the x values at that location. If
// the columns of A are not linearly independent or if the initial set is not
// feasible, valid is false.
func initializeFromBasic(ab *mat64.Dense, b []float64) (xb []float64, err error) {
	m, _ := ab.Dims()
	xb = make([]float64, m)
	xbMat := mat64.NewVector(m, xb)

	err = xbMat.SolveVec(ab, mat64.NewVector(m, b))
	if err != nil {
		return nil, errors.New("lp: subcolumns of A for supplied initial basic singular")
	}
	// The solve ensures that the equality constraints are met (ab * xb = b).
	// Thus, the solution is feasible if and only if all of the x's are positive.
	allPos := true
	for _, v := range xb {
		if v < -initPosTol {
			allPos = false
			break
		}
	}
	if !allPos {
		return xb, errors.New("lp: supplied subcolumns not a feasible solution")
	}
	return xb, nil
}

// extractColumns creates a new matrix out of the columns of A specified by cols.
func extractColumns(A mat64.Matrix, cols []int) *mat64.Dense {
	r, _ := A.Dims()
	sub := mat64.NewDense(r, len(cols), nil)
	col := make([]float64, r)
	for j, idx := range cols {
		mat64.Col(col, idx, A)
		sub.SetCol(j, col)
	}
	return sub
}

// findInitialBasic finds an initial basic solution.
func findInitialBasic(A mat64.Matrix, b []float64) ([]int, *mat64.Dense, []float64, error) {
	m, n := A.Dims()
	basicIdxs := findLinearlyIndependent(A)
	if len(basicIdxs) != m {
		return nil, nil, nil, ErrSingular
	}
	// It may be that this linearly independent basis is also a feasible set. If
	// so, the Phase I problem can be avoided. Check if it is.
	ab := extractColumns(A, basicIdxs)
	cond := mat64.Cond(ab, 1)
	if cond > 1e12 {
		panic("linearly dependend basic has bad condition number")
	}
	xb, err := initializeFromBasic(ab, b)
	if err == nil {
		// Initial problem feasible.
		return basicIdxs, ab, xb, nil
	}

	// This set was not feasible. Instead the "Phase I" problem must be solved
	// to find an initial feasible set of basis.
	//
	// Method: Construct an LP whose optimal solution is a feasible solution
	// to the original LP.
	// 1) Introduce an artificial variable x_{n+1}.
	// 2) Let x_j be the most negative element of x_b (largest constraint violation).
	// 3) Add the artificial variable to A with:
	//      a_{n+1} = b - \sum_{i in basicIdxs} a_i + a_j
	//    swap j with n+1 in the basicIdxs.
	// 4) Define a new LP:
	//   minimize  x_{n+1}
	//   subject to [A A_{n+1}][x_1 ... x_{n+1}] = b
	//          x, x_{n+1} >= 0
	// 5) Solve this LP. If x_{n+1} != 0, then the problem is infeasible, otherwise
	// the found basis can be used as an initial basis for phase II.
	//
	// The extra column in Step 3 is defined such that the vector of 1s is an
	// initial feasible solution.

	// Find the largest constraint violator.
	// Compute a_{n+1} = b - \sum{i in basicIdxs}a_i + a_j. j is in basicIDx, so
	// instead just subtract the basicIdx columns that are not minIDx.
	/*
		fmt.Println("ab = ", ab)
		fmt.Println("basicIdx", basicIdxs)
		fmt.Println("xb = ", xb)
	*/
	minIdx := floats.MinIdx(xb)
	aX1 := make([]float64, m)
	copy(aX1, b)
	col := make([]float64, m)
	for i, v := range basicIdxs {
		if i == minIdx {
			continue
		}
		mat64.Col(col, v, A)
		floats.Sub(aX1, col)
	}

	// Construct the new LP.
	// aNew = [A, a_{n+1}]
	// bNew = b
	// cNew = 1 for x_{n+1}
	aNew := mat64.NewDense(m, n+1, nil)
	aNew.Copy(A)
	aNew.SetCol(n, aX1)
	basicIdxs[minIdx] = n // swap minIdx with n in the basic set.
	c := make([]float64, n+1)
	c[n] = 1

	condANew := mat64.Cond(aNew, 1)
	if condANew > 1e12 {
		panic("initial simplex condition number bad")
	}

	/*
		// Validation code.
		// The vector of all 1s should be a feasible solution to this new LP
		aSharp := extractColumns(aNew, basicIdxs)

		var tmpSharp mat64.Vector
		ones := mat64.NewVector(m, nil)
		for i := 0; i < ones.Len(); i++ {
			ones.SetVec(i, 1)
		}
		tmpSharp.MulVec(aSharp, ones)
		if !floats.EqualApprox(tmpSharp.RawVector().Data, b, 1e-10) {
			panic("ones not feasible")
		}
	*/

	// Solve this linear program (but with an initial feasible solution provided this time).
	_, xOpt, newBasic, err := simplex(basicIdxs, c, aNew, b, 1e-10)
	if err != nil {
		return nil, nil, nil, errors.New(fmt.Sprintf("lp: error finding feasible basis: %s", err))
	}

	/*
		fmt.Println("Done phase I")
		fmt.Println("xopt = ", xOpt)
		fmt.Println("xopt n", xOpt[n])
	*/

	// If n+1 is part of the solution basis then the problem is infeasible. If
	// not, then the problem is feasible and newBasic is an initial feasible
	// solution.
	if math.Abs(xOpt[n]) > phaseIZeroTol {
		return nil, nil, nil, ErrInfeasible
	}

	// The value is zero. If it's not in the basis then the basis is a feasible
	// solution.
	basicIdx := -1
	basicMap := make(map[int]struct{})
	for i, v := range newBasic {
		if v == n {
			basicIdx = i
		}
		basicMap[v] = struct{}{}
		xb[i] = xOpt[v]
	}
	if basicIdx == -1 {
		// Not in the basis.
		ab = extractColumns(A, newBasic)
		return newBasic, ab, xb, nil
	}

	// The value is zero, but it's still in the basis. Try swapping in other
	// columns until a feasible solution is found.
	for i := range xOpt {
		if _, inBasic := basicMap[i]; inBasic {
			// Already in basis.
			continue
		}
		// Replace n with this column
		newBasic[basicIdx] = i
		ab := extractColumns(A, newBasic)
		xb, err := initializeFromBasic(ab, b)
		if err == nil {
			// Problem feasible with this column
			return newBasic, ab, xb, nil
		}
	}
	// Could not find feasible set.
	return nil, nil, nil, ErrInfeasible
	// Should just say it's infeasible rather than panic, but panic now for testing.
	//panic("lp: PhaseI returned feasible, but no feasible column set found.")
}

// findLinearlyIndependnt finds a set of linearly independent columns of A, and
// returns the column indexes of the linearly independent columns.
func findLinearlyIndependent(A mat64.Matrix) []int {
	m, n := A.Dims()
	idxs := make([]int, 0, m)
	columns := mat64.NewDense(m, m, nil)
	newCol := make([]float64, m)
	// Walk in reverse order because slack variables are typically the last columns
	// of A.
	for i := n - 1; i >= 0; i-- {
		if len(idxs) == m {
			break
		}
		mat64.Col(newCol, i, A)
		if len(idxs) == 0 {
			// A column is linearly independent from the null set.
			// This is what needs to be changed if zero columns are allowed, as
			// a column of all zeros is not linearly independent from itself.
			columns.SetCol(len(idxs), newCol)
			idxs = append(idxs, i)
			continue
		}
		if linearlyDependent(mat64.NewVector(m, newCol), columns.View(0, 0, m, len(idxs)), linDepTol) {
			continue
		}
		columns.SetCol(len(idxs), newCol)
		idxs = append(idxs, i)
	}
	return idxs
}

// linearlyDependent returns whether the vector is linearly dependent
// with the columns of A. It assumes that A is a full-rank matrix.
func linearlyDependent(vec *mat64.Vector, A mat64.Matrix, tol float64) bool {
	// Add vec to the columns of A, and see if the condition number is reasonable.
	m, n := A.Dims()
	aNew := mat64.NewDense(m, n+1, nil)
	aNew.Copy(A)
	col := mat64.Col(nil, 0, vec)
	aNew.SetCol(n, col)
	cond := mat64.Cond(aNew, 1)
	return cond > 1e12
}
