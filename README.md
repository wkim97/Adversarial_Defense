# Adversarial_Defense

1. apply_noise.py - Apply Gaussian and PGD noise
to MNIST images. Stores resulting images to 
"noisy_images".

2. noise_extraction.py - Extract noise from 
noisy images created from apply_noise.py. Stores
resulting noise to "noise".

3. train_gan.py - Model noise created from 
noise_extraction.py. Stores resulting noise 
models to "noise_models".

4. add_noise_to_clean - Apply noise models to 
clean images. Stores resulting noisy images to 
"generated_noisy_images".