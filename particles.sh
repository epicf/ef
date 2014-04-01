awk '/### Particles/ {flag=1; for (i = 1; i <= 2; i++) getline;} flag {print}' $@
