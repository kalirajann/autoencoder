import os
import numpy as np
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.losses import MeanSquaredError
from PIL import Image

# ----------------------------
# MODEL 2: MODIFIED AUTOENCODER (REDUCED LATENT SPACE)
# ----------------------------
# This model is identical to Model 1 except for the latent dimension.
# Reducing latent dimension forces the model to compress more information
# into fewer dimensions, leading to increased information loss and
# reconstruction artifacts (blurriness, loss of fine details).

def main():
    # 1. Load Fashion-MNIST dataset
    print("Loading Fashion-MNIST dataset...")
    (x_train, _), (x_test, _) = fashion_mnist.load_data()
    
    # Normalize pixel values to [0, 1]
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    
    # Flatten input images for dense autoencoder
    x_train_flat = x_train.reshape((-1, 28 * 28))
    x_test_flat = x_test.reshape((-1, 28 * 28))
    
    print(f"Training samples: {x_train_flat.shape[0]}")
    print(f"Test samples: {x_test_flat.shape[0]}")
    
    # 2. Build the autoencoder architecture (same as Model 1, but with reduced latent dimension)
    input_dim = 28 * 28
    # MODIFICATION: Reduced latent dimension from 64 to 32
    # This forces the bottleneck to compress 784 pixels into only 32 dimensions,
    # increasing information loss. The model must discard more details to fit
    # the essential information into the smaller space, resulting in blurrier
    # reconstructions with more artifacts.
    latent_dim = 32  # Reduced from 64 in Model 1
    
    # Encoder (architecture identical to Model 1)
    encoder_input = Input(shape=(input_dim,), name='encoder_input')
    x = Dense(256, activation='relu', name='enc_dense_1')(encoder_input)
    x = Dense(128, activation='relu', name='enc_dense_2')(x)
    x = Dense(64, activation='relu', name='enc_dense_3')(x)
    latent = Dense(latent_dim, activation='relu', name='latent')(x)
    
    encoder = Model(inputs=encoder_input, outputs=latent, name='encoder')
    
    # Decoder (architecture identical to Model 1)
    decoder_input = Input(shape=(latent_dim,), name='decoder_input')
    x = Dense(64, activation='relu', name='dec_dense_1')(decoder_input)
    x = Dense(128, activation='relu', name='dec_dense_2')(x)
    x = Dense(256, activation='relu', name='dec_dense_3')(x)
    decoder_output = Dense(input_dim, activation='sigmoid', name='decoder_output')(x)
    
    decoder = Model(inputs=decoder_input, outputs=decoder_output, name='decoder')
    
    # Autoencoder: encoder + decoder
    ae_input = encoder_input
    ae_output = decoder(encoder(ae_input))
    autoencoder = Model(inputs=ae_input, outputs=ae_output, name='autoencoder')
    
    # 3. Compile model (same optimizer and loss as Model 1)
    optimizer = RMSprop(learning_rate=0.001)  # same as Model 1
    loss_fn = MeanSquaredError()  # reconstruction loss
    autoencoder.compile(optimizer=optimizer, loss=loss_fn)
    
    # 4. Print summaries
    print("\n" + "=" * 50)
    print("Encoder Summary:")
    print("=" * 50)
    encoder.summary()
    
    print("\n" + "=" * 50)
    print("Decoder Summary:")
    print("=" * 50)
    decoder.summary()
    
    print("\n" + "=" * 50)
    print("Autoencoder Summary:")
    print("=" * 50)
    autoencoder.summary()
    
    # 5. Train the model (same epochs and batch size as Model 1)
    epochs = 20  # same as Model 1
    batch_size = 128  # same as Model 1
    
    print("\n" + "=" * 50)
    print("Training Autoencoder...")
    print("=" * 50)
    
    history = autoencoder.fit(
        x_train_flat, x_train_flat,
        epochs=epochs,
        batch_size=batch_size,
        shuffle=True,
        validation_data=(x_test_flat, x_test_flat),
        verbose=1
    )
    
    # Print training loss per epoch
    print("\n" + "=" * 50)
    print("Training Loss per Epoch:")
    print("=" * 50)
    for epoch, loss in enumerate(history.history['loss'], start=1):
        print(f"Epoch {epoch}: {loss:.6f}")
    
    # 6. Final training loss
    final_train_loss = history.history['loss'][-1]
    
    # 7. Evaluate on test set
    print("\n" + "=" * 50)
    print("Evaluating on test set...")
    print("=" * 50)
    test_loss = autoencoder.evaluate(x_test_flat, x_test_flat, batch_size=batch_size, verbose=1)
    
    # 8. Reconstruct test images (using same indices as Model 1)
    print("\n" + "=" * 50)
    print("Reconstructing test images...")
    print("=" * 50)
    
    # Select exactly 5 test images (same indices as Model 1: 0, 1, 2, 3, 4)
    n_images = 5
    indices = [0, 1, 2, 3, 4]
    test_samples = x_test_flat[indices]
    reconstructed = autoencoder.predict(test_samples, verbose=0)
    
    # Reshape for saving
    originals = test_samples.reshape((n_images, 28, 28))
    recos = reconstructed.reshape((n_images, 28, 28))
    
    # Ensure output directory exists
    output_dir = "reconstructions"
    os.makedirs(output_dir, exist_ok=True)
    
    print("\nSelected test image indices:")
    for idx in indices:
        print(f"  Index: {idx}")
    
    # Save images with Model 2 naming convention
    # Note: We expect more reconstruction artifacts due to reduced latent space
    print("\nSaving images...")
    for i in range(n_images):
        # Save reconstructed images with Model 2 naming
        reco_path = os.path.join(output_dir, f"reconstructed_model2_{i+1}.png")
        
        # Reconstructed image (with expected artifacts from information compression)
        img2 = (np.clip(recos[i], 0, 1) * 255).astype('uint8')
        Image.fromarray(img2, mode='L').save(reco_path)
        
        print(f"  Saved: {reco_path}")
    
    # 9. Print final model outputs
    print("\n----------------------------------------------")
    print("MODEL 2: MODIFIED AUTOENCODER (REDUCED LATENT SPACE)")
    print("----------------------------------------------")
    print(f"Latent Dimension Used: {latent_dim}")
    print(f"Final Training Loss: {final_train_loss:.6f}")
    print(f"Final Test Reconstruction Loss: {test_loss:.6f}")


if __name__ == "__main__":
    main()

