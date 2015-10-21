package optimize

import (
	"math"
	"time"

	"github.com/gonum/matrix/mat64"
)

// Global uses a global  seeks to find the gloabl minimum of a minimization problem.
// A maximization problem can be transformed into a minimization problem by
// multiplying the function by -1.
//
// Global is different than local because it tries to find the global minimum,
// which can be anywhere. The only reasonable convergence metric is the updates
// on major iteration. GradientThreshold is not checked during convergence, and
// initial values are not used.
//
// TODO(btracey): Split into LocalSettings and GlobalSettings with a common Settings
// struct that is embedded? In initializeOptimization
// TODO(btracey): LocalMethod / GlobalMethod or Method and GlobalMethod?
func Global(p Problem, dim int, settings *Settings, method GlobalMethod) (*Result, error) {
	if method == nil {
		// use getDefaultMethod with wrapper
		panic("need method")
	}
	if settings == nil {
		settings = DefaultSettings()
	}
	startTime, err := initializeOptimization(&p, settings, method)
	if err != nil {
		return nil, err
	}
	if settings.FunctionConverge == nil {
		return nil, ErrFunctionConvergeNil
	}
	stats := &Stats{}
	optLoc := &Location{
		F: math.Inf(1),
		X: make([]float64, dim),
	}
	if method.Needs().Gradient {
		optLoc.Gradient = make([]float64, dim)
	}
	if method.Needs().Hessian {
		optLoc.Hessian = mat64.NewSymDense(dim, nil)
	}

	status, err := minimizeGlobal(p, method, settings, stats, optLoc, startTime)

	if settings.Recorder != nil && err == nil {
		// Send the optimal location to Recorder.
		err = settings.Recorder.Record(optLoc, PostIteration, stats)
	}
	stats.Runtime = time.Since(startTime)
	return &Result{
		Location: *optLoc,
		Stats:    *stats,
		Status:   status,
	}, err
}

func minimizeGlobal(p Problem, method GlobalMethod, settings *Settings, stats *Stats, optLoc *Location, startTime time.Time) (status Status, err error) {
	concurrent := settings.Concurrent
	if concurrent == 0 {
		concurrent = 1
	}

	jobs := make(chan LocationSender)
	quit := make(chan struct{})
	updater := make(chan LocationSender, settings.Concurrent)
	closer := make(chan struct{}, settings.Concurrent)

	go func() {
		method.Schedule(len(optLoc.X), concurrent, jobs, quit)
	}()

	// High level: Method continually sends locations and operations on the
	// jobs channel. These are received by the workers, who evaluate the function,
	// and send the result to the updater. The updater updates optLoc if necessary
	// and tests for convergence.
	// On convergence, globalUpdater closes the quit channel. It the the methods
	// job to stop sending values and close the jobs channel. Worker sees the jobs
	// channel has closed, and signaled it has completed. When all of the workers
	// have completed, the updater channel is closed so the updater knows to return.

	// Local has the property that the Method keeps feeding locations until done.
	// This is an important property.

	// Launch worker goroutines
	for i := 0; i < concurrent; i++ {
		go globalWorker(p, jobs, updater, closer)
	}

	// This closes the updater channel once all of the workers have quit. This
	// then causes updater to quit.
	go func() {
		for i := 0; i < concurrent; i++ {
			<-closer
		}
		close(updater)
	}()

	// Keep updating until quit
	status = globalUpdater(stats, settings, updater, quit, optLoc, startTime)
	return status, nil
}

// Worker receives a location and an operation. Worker sends the result of that
// operation to globalUpdater (as stats updates and checking for convergence
// must happen in serial). Once the needed information from Location is used,
// it sends the location back to the GlobalMethod so it can use the result.

// either we have to copy location or we have to deal with the fact this could
// be a bottle neck.
func globalUpdater(stats *Stats, settings *Settings, updater <-chan LocationSender, quit chan struct{}, optLoc *Location, startTime time.Time) Status {
	first := true
	var status Status
	var closed bool
	for update := range updater {
		// Get all of the needed information on the global update so can let
		// the worker continue as fast as possible
		eval := update.Operation
		var f float64
		if eval == MajorIteration {
			f = update.Location.F
			if f < optLoc.F {
				copyLocation(optLoc, update.Location)
			}
		}
		update.Evaluated <- struct{}{}

		// TODO(btracey): Check that it's a valid operation
		if eval == MajorIteration {
			if first {
				settings.FunctionConverge.Init(f)
				first = false
			} else {
				status = settings.FunctionConverge.FunctionConverged(f)
			}
		} else {
			if eval&FuncEvaluation != 0 {
				stats.FuncEvaluations++
			}
			if eval&GradEvaluation != 0 {
				stats.GradEvaluations++
			}
			if eval&HessEvaluation != 0 {
				stats.HessEvaluations++
			}
		}

		// TODO(btracey): Add statuser check for both Problem and Optimize.
		settings.Runtime = time.Since(startTime)
		if status == NotTerminated {
			if optLoc.F < settings.FunctionThreshold {
				status = FunctionThreshold
			} else {
				status = checkLimits(optLoc, stats, settings)
			}
		}
		if !closed && status != NotTerminated {
			closed = true
			close(quit)
		}
	}
	return status
}

// globalWorker receives a location and operation to evaluate from the global
// queue, evaluates the function, and sends the result to the global updater.
// OptLoc is ONLY to keep the sizing code in one place. Location is temporary
// storage.
// Method's job to close queue once quit happens
func globalWorker(p Problem, jobs <-chan LocationSender, updater chan<- LocationSender, closer chan<- struct{}) {
	var x []float64
	for job := range jobs {
		eval := job.Operation
		if x == nil {
			x = make([]float64, len(job.Location.X))
		}
		copy(x, job.Location.X)
		if eval&FuncEvaluation != 0 {
			job.Location.F = p.Func(x)
		}
		if eval&GradEvaluation != 0 {
			p.Grad(job.Location.Gradient, x)
		}
		if eval&HessEvaluation != 0 {
			p.Hess(job.Location.Hessian, x)
		}
		updater <- job
	}
	// Communicate that has finished
	closer <- struct{}{}
}

func initializeOptimization(p *Problem, settings *Settings, method GlobalMethod) (start time.Time, err error) {
	if p.Func == nil {
		panic(badProblem)
	}
	startTime := time.Now()
	if err := p.satisfies(method); err != nil {
		return startTime, err
	}
	if p.Status != nil {
		_, err := p.Status()
		if err != nil {
			return startTime, err
		}
	}
	if settings.Recorder != nil {
		// Initialize Recorder first. If it fails, we avoid the (possibly
		// time-consuming) evaluation at the starting location.
		err := settings.Recorder.Init()
		if err != nil {
			return startTime, err
		}
	}
	return startTime, nil
}

func DefaultSettingsGlobal() *Settings {
	return &Settings{
		FunctionThreshold: math.Inf(-1),
		FunctionConverge: &FunctionConverge{
			Absolute:   1e-10,
			Iterations: 20,
		},
	}
}

// LocationSender sends a location to be evaluated and its iteration.
// The location with values populated will be returned on the receive channel.
type LocationSender struct {
	Location  *Location
	Operation Operation
	Evaluated chan<- struct{}
}

// Init initializes the GlobalMethod. It continues to send new locations on send
// until a message from quit is sent.
// Method must close jobs when a receive from quit happens
type GlobalMethod interface {
	Schedule(dim, concurrent int, jobs chan<- LocationSender, quit <-chan struct{})

	Needser
}
