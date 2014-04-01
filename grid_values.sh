awk '/### Grid/ {flag=1; for (i = 1; i <= 4; i++) getline;} /### Particles/{flag=0} flag {print}' $@
