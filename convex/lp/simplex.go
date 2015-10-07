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
	linDepTol  = 1e-10
	initPosTol = 1e-14 // tolerance on x being positive for the initial feasible.
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
		ab := extractColumns(A, initialBasic)
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

	// Find an initial set of basic vectors and an initial
	// basic solution. The initial basic solution should be feasible and contain
	// a set of linearly independent columns of A.

	// First, re-arrange the variables such that the last m columns are linearly
	// independent
	// A = [A_n A_B]  A_B = mxm, A_n = m x (n-m)
	// x = [x_n; x_b] x_b = m  x_n = n-m
	// A x = b
	// A_n x_n + A_b x_b = b
	// If we set x_n = 0, then A_b x_b = b --> x_b = A_b^-1 b
	// x has at least n-m zero elements --> "Basic solution"
	// x_n nonbasic variables
	// x_b basic variables
	// More generally, just store a list of the basic variables.

	// TODO(btracey): If we only need ab^T and An^T, we can work row-wise helping
	// caches.

	// fmt.Println("a =", A)
	// fmt.Println("b =", b)
	//fmt.Println("c = ", c)

	//fmt.Printf("a orig format\n% 0.4v\n", mat64.Formatted(A))
	//fmt.Printf("a orig = %#v\n", A)
	//fmt.Printf("b orig %#v\n", b)
	//fmt.Printf("c orig %#v\n ", c)
	/*
		isZero := true
		for _, v := range c {
			if v != 0 {
				isZero = false
			}
		}
		if isZero {
			// c all zeros. Need to find an initial basic
			//return 0, make([]float64, , nil, nil
		}
	*/

	//fmt.Println("at start")
	//fmt.Println("ab = ", ab)
	// fmt.Println("xb = ", xb)

	// Verify sizes
	// TODO(btracey): remove when we're sure the code is correct.
	if len(basicIdxs) != m {
		panic("lp: unexpected bad idx size")
	}
	if len(xb) != m {
		panic("lp: unexpected bad xb size")
	}
	abr, abc := ab.Dims()
	if abr != m {
		panic("lp: unexpected bad ab rows")
	}
	if abc != m {
		panic("lp: unexpected bad ab cols")
	}

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
	_ = bVec
	cbVec := mat64.NewVector(len(cb), cb)
	//cnVec := mat64.NewVector(len(cn), cn)
	// Phase 2: Solve the simplex.
	// We have a basic feasible set. basicIdxs contain the non-zero x elements,
	// aBasic contains the columns of A that correspond, and xb contains the
	// non-zero elements of the feasible solution.

	//abLU := &mat64.LU{}
	//abLU.Factorize(ab)
	_ = xb
	r := make([]float64, n-m)
	aCol := mat64.NewVector(m, nil)
	move := make([]float64, m)
	lastCost := math.Inf(1)
	// fmt.Println("Starting simplex for loop")
	for {
		fmt.Println(basicIdxs)
		// Compute the reduced costs.
		// r = cn - an^T ab^-T cb
		var tmpMat mat64.Dense
		err := tmpMat.Solve(ab.T(), cbVec)
		//abt := mat64.DenseCopyOf(ab.T())
		//err := simplexSolve(&tmpMat, abt, cbVec)
		if err != nil {
			fmt.Println("ab^T = ", ab)
			fmt.Println("err = ", err)
			panic("lp: unexpected linear solve error")
		}
		tmpVec2 := mat64.NewVector(m, mat64.Col(nil, 0, &tmpMat))
		//tmpVec2 := mat64.NewVector(m, mat64.Col(nil, 0, tmpMat))
		tmpVec := mat64.NewVector(n-m, nil)
		tmpVec.MulVec(an.T(), tmpVec2)
		floats.SubTo(r, cn, tmpVec.RawVector().Data)

		bland := false
		var minIdx, replace int
		var done bool
		// fmt.Println("r = ", r)
		// fmt.Println("move =", move)
		// fmt.Println("ab = ", ab)
		// fmt.Println("nonbasic = ", nonBasicIdx)
		minIdx, replace, done, err = findNext(move, aCol, bland, r, tol, ab, xb, nonBasicIdx, A)
		if done {
			break
		}
		if err != nil {
			return math.Inf(-1), nil, nil, err
		}

		if move[replace] == 0 {
			// Degeneracy is when at least one i in the BFS is equal to zero.
			// Happens when two BFSs overlap.
			// Instead of choosing the minimum index of r, we need to choose the
			// smallest index of r that is negative. Then recompute move, and then
			// take the smallest variable in the index of move. Needs to be smallest
			// index as per row of A.
			bland := true
			minIdx, replace, done, err = findNext(move, aCol, bland, r, tol, ab, xb, nonBasicIdx, A)
			// Shouldn't be done or err here
			if done {
				panic("lp: bad done")
			}
			if err != nil {
				return math.Inf(-1), nil, nil, err
			}
			/*
				if move[replace] == 0 {
					panic("lp: move still zero")
				}
			*/
		}
		basicIdxs[replace], nonBasicIdx[minIdx] = nonBasicIdx[minIdx], basicIdxs[replace]
		cb[replace], cn[minIdx] = cn[minIdx], cb[replace]
		// Replace columns as well
		tmp1 := mat64.Col(nil, minIdx, an)
		tmp2 := mat64.Col(nil, replace, ab)
		//tmp1 := an.Col(nil, minIdx)
		//tmp2 := ab.Col(nil, replace)
		an.SetCol(minIdx, tmp2)
		ab.SetCol(replace, tmp1)

		abshare := extractColumns(A, basicIdxs)
		fmt.Println("abshare same")
		fmt.Println(mat64.Equal(abshare, ab))
		fmt.Println(basicIdxs)
		//fmt.Println(A)
		//fmt.Println(ab)
		fmt.Printf("a orig format\n% 0.4v\n", mat64.Formatted(A))
		fmt.Printf("ab format\n% 0.4v\n", mat64.Formatted(ab))

		var xbVec mat64.Dense
		err = xbVec.Solve(ab, bVec)
		//err = simplexSolve(&xbVec, ab, bVec)
		if err != nil {
			fmt.Println("ab = ", ab)
			fmt.Println("err = ", err)
			panic("lp: unexpected linear solve error")
		}
		//xbVec.Col(xb, 0)
		mat64.Col(xb, 0, &xbVec)
		cost := floats.Dot(cb, xb)
		if cost-lastCost > 1e-10 {
			fmt.Println("cost = ", cost)
			fmt.Println("lastCost = ", lastCost)
			panic("lp: cost should never increase")
		}
		lastCost = cost
	}
	opt := floats.Dot(cb, xb)
	// All non-basic variables are zero.
	xopt := make([]float64, n)
	for i, v := range basicIdxs {
		xopt[v] = xb[i]
	}
	return opt, xopt, basicIdxs, nil

	// TODO(btracey): Need to see if x_b is a basic feasible solution.

	// At solution:
	// 1) Feasible region has vertices where m constraints intersect
	// 2] We can always find an optimum at a vertex

	// Need A to be full rank (otherwise answer is inf)
	// If a feasible solution exists, then a basic feasible solution exists
	// If an optimal feasible solution exists, then an optimal basic feasible
	// solution exists
	// This implies that we only need to look at basic feasible solutions.
	// Simplex tries to find the best basic feasible solution by replacing one
	// basic variable with a non-basic one

	//for {
	// xk is the basic x at step k
	// If xn is non-zero
	// xb = Ab^-1 b - Ab^-1 An xn
	// Cost is  c^T x = cb^T xb + cn^T xn
	// = cb^T Ab^-1 b + (cn^T - cb^T Ab^-1 An) xn   (cost expressed just in xn)
	// = c^T xk + r^T xn (first term is cost at k)
	// From non-negativity constraints, xn can only increase.
	// r quantifies how much each nonbasic variable affects the cost.
	// If r >= 0, then no improvement to the objective is possible, and x_k is
	// optimal.
	// Otherwise, pick the most negative R and choose to increase non-basic
	// variable x_e  (entering variable)
	// Remove the x_b that goes to zero first when x_e is increased.
	// xb = Ab^-1 b - Ab^-1 An xn
	//    = Ab^-1 b - Ab^-1 Ae xe
	//    = bhat + d x_e
	//  xe = bhat_i / - d_i
	// Interested in the first basic variable for which this is true, so
	// minimum over i (assuming d is negative).
	// If no d_i < 0, then it implies that LP is unbounded.

	// Trickiness if b == 0 as x_e can't be increased at all, so objective
	// is not decreased.

	// TODO(btracey): Look at duality gap for tolerance?
	//}
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
	ab := extractColumns(A, initialBasic)
	xb, err = initializeFromBasic(ab, b)
	if err != nil {
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
	_, xOpt, newBasic, err := simplex(basicIdxs, c, aNew, b, 1e-14)
	if err != nil {
		return nil, nil, nil, errors.New(fmt.Sprintf("lp: error finding feasible basis: %s", err))
	}

	// If n+1 is part of the solution basis then the problem is infeasible. If
	// not, then the problem is feasible and newBasic is an initial feasible
	// solution.
	var inBasis bool
	for i, v := range newBasic {
		if v == n {
			inBasis = true
			break
		}
		xb[i] = xOpt[v]
	}
	if inBasis {
		return nil, nil, nil, ErrInfeasible
	}
	ab = extractColumns(A, newBasic)
	return newBasic, ab, xb, nil
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
		if len(idxs) == m {
			break
		}
	}
	return idxs
}

// linearlyDependent returns whether the vector is linearly dependent
// with the columns of A. It assumes that A is a full-rank matrix.
func linearlyDependent(vec *mat64.Vector, A mat64.Matrix, tol float64) bool {
	// A vector is linearly dependent on the others if it can
	// be computed from a weighted sum of the existing columns, that
	// is c_new = \sum_i w_i c_i. In matrix form, this is represented
	// as c_new = C * w, where C is the composition of the existing
	// columns. We can solve this system of equations for w to get w^.
	// If C * w^ = c_new, then c_new is linearly dependent. Otherwise
	// it is independent.

	var wHat mat64.Vector
	err := wHat.SolveVec(A, vec)
	if err != nil {
		// Solve can only fail if C is not of full rank. Method assumes A is
		// of full rank.
		panic("lp: unexpected linear solve failure")
	}
	var test mat64.Vector
	test.MulVec(A, &wHat)
	return mat64.EqualApprox(&test, vec, linDepTol)
}

// move stored in place
func findNext(move []float64, aCol *mat64.Vector, bland bool, r []float64, tol float64, ab *mat64.Dense, xb []float64, nonBasicIdx []int, A mat64.Matrix) (minIdx, replace int, done bool, err error) {
	m, _ := A.Dims()
	// Find the element with the minimum reduced cost.
	if bland {
		fmt.Println("in bland")
		// Find the first negative entry of r.
		// TODO(btracey): Is there a way to communicate entries that are supposed
		// to be zero? Should we round all numbers below a tol to zero.
		// Don't overload the solution tolerance with floating point error
		// tolerance.

		// TODO(btracey); Should only replace if the swapped row keeps aCol
		// full rank.
		var found bool
		for i, v := range r {
			negTol := 1e-14
			// Zero column can cause this replacement to be singular. Correct
			// replacing may be able to deal with that issue.
			if v < -negTol {
				minIdx = i
				found = true
				break
			}

		}
		if !found {
			panic("lp beale: no negative argument found")
		}
	} else {
		// Replace the most negative element in the simplex.
		minIdx = floats.MinIdx(r)
	}

	// If there are no negative entries, then we have found an optimal
	// solution.
	if !bland && r[minIdx] >= -tol {
		// Found minimum successfully
		// fmt.Println("found successfully")
		return -1, -1, true, nil
	}
	// fmt.Println("not found successfully")
	bHat := xb // ab^-1 b
	// fmt.Println("bhat = ", bHat)
	// fmt.Println(ab)
	colIdx := nonBasicIdx[minIdx]
	// TODO(btracey): Can make this a column view.
	for i := 0; i < m; i++ {
		aCol.SetVec(i, A.At(i, colIdx))
	}
	// d = -ab^-1 * A_minidx.
	var dVec mat64.Dense
	err = dVec.Solve(ab, aCol)
	if err != nil {
		panic("lp: unexpected linear solve error")
	}
	d := mat64.Col(nil, 0, &dVec)
	//d := dVec.Col(nil, 0)
	floats.Scale(-1, d)

	// If no di < 0, then problem is unbounded.
	if floats.Min(d) >= 0 {
		// fmt.Printf("abmat =\n%0.4v\n", mat64.Formatted(ab))
		// fmt.Println("ab = ", ab)
		// fmt.Println("aCol = ", aCol)
		// fmt.Println("Unbounded, d =", d)
		// Problem is unbounded
		// TODO(btracey): What should we return
		return -1, -1, false, ErrUnbounded
	} else {
		// fmt.Println("not unbounded")
	}
	//fmt.Println("bhat", bHat)
	//fmt.Println("d = ", d)
	for i, v := range d {
		// Only look at the postive d values
		if v >= 0 {
			move[i] = math.Inf(1)
			continue
		}
		move[i] = bHat[i] / -v
	}
	//fmt.Println("move", move)
	// Replace the smallest movement in the basis.
	fmt.Println(move)
	replace = floats.MinIdx(move)
	return minIdx, replace, false, nil
}
