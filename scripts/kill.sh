#!/bin/bash

# Help function
Help()
{
   echo "Script to kill scripts on a tpu pod slice. Needs gcloud to be logged in."
   echo
   echo "Syntax: kill.sh [n|z|c|h]"
   echo "options:"
   echo "n     TPU VM pod slice name in GCS."
   echo "z     TPU VM zone in GCS."
   echo "c     TPU VM count of hosts in pod slice."
   echo "h     Print this Help."
   echo
}

# Get args
while getopts "n:z:c:h" option;
do
   case $option in
      n)
         NAME=$OPTARG;;
      z)
         ZONE=$OPTARG;;
      c)
         COUNT=$OPTARG;;
      h)
         Help
         exit;;
      \?) # Invalid option
         echo "Error: invalid option. Use -h to see help."
   esac
done


for ((i=0;i<COUNT;i++));
do
    PID=$(gcloud compute tpus tpu-vm ssh "$NAME" \
        --zone "$ZONE" \
        --worker="$i" \
        --command="pgrep -f 'python3 scripts/launch.py'"
    );
    if [ "$PID" != "" ];
    then
        gcloud compute tpus tpu-vm ssh "$NAME" \
            --zone "$ZONE" \
            --worker="$i" \
            --command="kill -n 9 $PID";
    fi;
done;
