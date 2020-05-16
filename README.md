[![Build Status](https://travis-ci.com/ricardolira/mpfa-d.svg?branch=master)](https://travis-ci.com/ricardolira/mpfa-d)
# mpfa-d


A MultiPoint Flux Approximation with diamond shape stencil for the solution of
two-phase fluid-flow problems.

Run it through docker-compose:

`$ docker-compose run test`


Your cases can be run the benchamrk cases with:

`$ docker-compose run benchmarks`

Add any benchmarks to your solver, importing from the `single_phase_cases` folder.

If you want to add any dependecies to your python code, do it via requirements and rebuild your image:

`$ docker-compose -f docker-compose.yml build`
