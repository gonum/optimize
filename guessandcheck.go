package optimize

import "sync"

// TODO(btracey): Add Rander interface to distmv.
type Rander interface {
	Rand([]float64) []float64
}

// GuessAndCheck is a global optimizer that evaluates the function at random
// locations. Not a good optimizer, but useful for comparison and debugging.
type GuessAndCheck struct {
	Rander Rander
}

func (g GuessAndCheck) Needs() struct{ Gradient, Hessian bool } {
	return struct{ Gradient, Hessian bool }{false, false}
}

func (g GuessAndCheck) Schedule(dim, concurrent int, jobs chan<- LocationSender, quit <-chan struct{}) {
	var wg sync.WaitGroup
	wg.Add(concurrent)
	for i := 0; i < concurrent; i++ {
		go func() {
			defer wg.Done()
			done := make(chan struct{})
			location := &Location{
				X: make([]float64, dim),
			}
			for {
				g.Rander.Rand(location.X)

				// See if it's time to quit
				select {
				case <-quit:
					return
				default:
				}

				// Evaluate the function and then send a major iteration update.
				jobs <- LocationSender{
					Operation: FuncEvaluation,
					Location:  location,
					Evaluated: done,
				}
				<-done

				jobs <- LocationSender{
					Operation: MajorIteration,
					Location:  location,
					Evaluated: done,
				}
				<-done
			}
		}()
	}
	wg.Wait()
	close(jobs)
}
