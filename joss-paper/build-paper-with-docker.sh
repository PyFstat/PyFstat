#!/usr/bin/env bash
docker run --rm \
    --volume $PWD/joss-paper:/data \
    --user $(id -u):$(id -g) \
    --env JOURNAL=joss \
    openjournals/paperdraft
