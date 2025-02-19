# **Deep Reinforcement Learning**  

## **Introduction**  
In this Lab I implented the **REINFORCE** and **DQN** algorithm to solve the **CartPole-v1** and the **Lunar-Lander-v3** environments using deep reinforcement learning. 

## **Dependencies**  
Make sure you have the following libraries installed before running the code:  

```bash
pip install torch gymnasium[box2d] matplotlib numpy gymnasium[other]
```

- **gymnasium**: For the CartPole and Lunar_Lander simulation environment  
- **torch**: To define and train neural networks  
- **matplotlib**: To visualize results  
- **numpy**: For numerical computations  

## **Running the Training**  
To train the model, run:  

```bash
python name_script.py train
```

Once trained, you can test the model using:  

```bash
python name_script.py test
```

where **name_script** can be 'cartpole' or 'LunarReinforce' 
<br><br>

## **REINFORCE**

### **CARTPOLE**  
The goal is to balance a pole on a moving cart for as long as possible by learning an optimal policy through policy gradients.The algorithm is considered solved when it achieves an average greater than 195 over 100 consecutive episodes.
This environment can be easily solved using the REINFORCE algorithm with well-chosen hyperparameters for the policy network and optimizer. In this case, I achieved rewards close to 8000 in under 200 episodes.
<br><br>
![Training Results](Ex1/Results/Cart.png)
<br><br>
![gif cartpol test](videos/cartpole.gif)

### **LUNAR LANDER**
To solve the LunarLander environment, the agent needs to achieve a score of 200 or higher for approximately 100 consecutive episodes.The REINFORCE algorithm leads to highly unstable results with significant variance in the obtained scores. To mitigate this issue, I added an extra layer to the policy network, reduced the gamma parameter to 0.95, introduced entropy regularization, and applied gradient clipping. However, these adjustments did not yield significant improvements—catastrophic forgetting remains a problem, and the average score still hovers around 20 points.
<br><br>
![Training Results](Ex1/Results/Lunar.png)
<br><br>
![gif lunar test](videos/lunar_reinforce.gif)
<br><br>

## **DQN**
To successfully solve the Lunar Lander environment, I implemented a Deep Q-Network (DQN) with a replay buffer   and update the network every four iterations to reduce variance. Epsilon decay was used to allow the agent to explore different alternatives, especially in the early stages of training. Also the use of Huber loss proved beneficial in improving stability. Additionally, I applied gradient clipping to further stabilize the training process.

The resulting model performs very well on this task, consistently solving the environment in approximately 160–180 episodes. Below, I have included the scores obtained over the 100 'winning episodes.'
<br><br>
![Training Results](Ex1/Results/Lunar.png)
<br><br>
And here is the final landing...
![gif lunar test](videos/lunar_dqn.gif)

## **License**  
This project is released under the MIT License.
