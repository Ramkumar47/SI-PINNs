# SIPINN parameter estimation CD with indieNet
SIPINN with smoothed data. With MSE as loss function and the V network
is trained till MSE = 0 ...

logs:
- It gave much better results directly from the model itself.

output:

PINN output

|-------------|-----------------|--------------|------------------|
| coefficient | predicted value | actual value | error percentage |
|-------------|-----------------|--------------|------------------|
| CD_0        | 0.03604919      | 0.036        | 0.1366           |
| CD_alpha    | 0.04058768      | 0.041        | 1.00565          |
| CD_dele     | 0.02583807      | 0.024        | 0.6228           |
|-------------|-----------------|--------------|------------------|


LS output

|-------------|-----------------|--------------|------------------|
| coefficient | predicted value | actual value | error percentage |
|-------------|-----------------|--------------|------------------|
| CD_0        | 0.03601820      | 0.036        | 0.0505           |
| CD_alpha    | 0.04080074      | 0.041        | 0.4860           |
| CD_dele     | 0.02575282      | 0.024        | 0.9506           |
|-------------|-----------------|--------------|------------------|
