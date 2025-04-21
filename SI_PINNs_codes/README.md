# SIPINN parameter estimation
Here, the delta_e values are smoothed to prevent any wiggles to be present
as the wiggle noise is not captured by the model and were not accurate.

It is to see if the model can back-calculate the coefficients with autodiff
with smooth data.

log
- the model works good, but the equation is not sensitive to changes in alpha, only
that term is having error of about 3%, others are all in the range of < 1%.
