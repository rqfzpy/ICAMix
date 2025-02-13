import torch

def mix(x, y, num_class, lamb):
    # Initialize an empty tensor for mixed images
    img_magnitudemix = torch.zeros_like(x)
    
    # Iterate over each class
    for i in range(num_class):
        # Extract data corresponding to the current class
        class_data = x[y == i]
        
        # Find indices where the labels match the current class
        indices = torch.where(y == i)
        
        # Skip if no data is found for the current class
        if class_data.shape[0] == 0:
            continue
        
        # Compute the FFT of the class data
        fft_feature = torch.fft.rfftn(class_data, dim=[2, 3])
        
        # Compute the phase of the FFT
        fft_pha = torch.angle(fft_feature)
        
        # Compute the mixed FFT amplitude
        mixed_fft_amp = torch.mean(torch.abs(fft_feature), dim=0, keepdim=True).repeat(fft_feature.shape[0], 1, 1, 1)
        
        # Combine the mixed FFT amplitude with the original phase
        mixed_fft_feature = mixed_fft_amp * torch.exp(1j * fft_pha)
        
        # Compute the inverse FFT to obtain the mixed data
        mixed_data = torch.fft.irfftn(mixed_fft_feature, s=(x.shape[2], x.shape[3]), dim=[2, 3])
        
        # Update the mixed image tensor with the mixed data for the current class
        img_magnitudemix[indices] = mixed_data 
    
    # Add the original image multiplied by lambda to the mixed image
    mixed_x = lamb * x + (1 - lamb) * img_magnitudemix
    
    # Return the mixed image
    return mixed_x