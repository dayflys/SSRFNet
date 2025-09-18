sudo nvidia-docker run -it --rm --ipc=host --shm-size 20G -v \
 $data_path:/data -v \
 $environment_path:/environment -v \
 $results_path:/results \
 env202308:latest