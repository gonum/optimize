package optimize

/*
// RestartLocal performs global optimization through random-restart local
// optimization.
// TODO(btracey): Some way of making Method non-nil? Otherwise can't do
// &RestartLocal{}
type RestartLocal struct {
	Rander         Rander
	MethodFactory  func() Method    // Method for the local optimization
	SettingFactory func() *Settings // Settings for the local optimization
}

func (r *RestartLocal) Needs() {
	return r.Method.Needs()
}

func (r *RestartLocal) Schedule(dim, concurrent int, jobs chan<- LocationSender, quit chan<- struct{}) {
	// Launch concurrent local optimizers
	for i := 0; i < concurrent; i++ {
		go func() {
			initialize := true
			method := r.MethodFactory()

			var location Location
			location.X = make([]float64, dim)
			if method.Needs().Gradient {
				location.Gradient = make([]float64, dim)
			}
			if method.Needs().Hessian {
				location.Hessian = mat64.NewSymDense(dim, nil)
			}
			evaluated := make(chan struct{})

			locationSend := LocationSender{
				Location:  location,
				Evaluated: evaluated,
			}

			for {
				if initialize {
					// Starting interation of an optimization
					// Use rander to find an initialX
					r.Rander.Rand(initX)
					if method.Needs().Gradient {
						locationSend.Operation = GradEvaluation
						jobs <- locationSend

					}

					stats := &Stats{}
					settings := r.SettingFactory()
					method.Init(loc)
				}
			}
		}()
	}
}
*/
