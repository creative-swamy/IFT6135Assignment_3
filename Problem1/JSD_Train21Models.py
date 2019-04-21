import JSD_Final as problem1
from matplotlib import pyplot as plt
import samplers
theta_initial = -1
theta_incremental = 0.1
number_of_models = 21
p_distribution = samplers.distribution1(0, 512)
JSD_Values = []
Theta_Values = []

for i in range(number_of_models):
    print("#####################", "Itteration",i+1, "######################################")
    q_distribution = samplers.distribution1(theta_initial, 512)
    JSD_M = problem1.JSD(hidden_size=64, mini_batch=512, learning_rate=0.0001, num_epochs=1000, print_interval=100)
    JSD_Values.append(JSD_M.run_main_loop(p_distribution, q_distribution))
    Theta_Values.append(theta_initial)
    theta_initial = theta_initial + theta_incremental
    
plt.plot(Theta_Values, JSD_Values, label='JSD')
plt.legend()
plt.show()
print( "Training complete" )