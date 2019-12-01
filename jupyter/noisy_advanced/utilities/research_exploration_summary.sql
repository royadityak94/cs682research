CREATE TABLE IF NOT EXISTS cs682research.research_exploration (
optimizer VARCHAR(255) NOT NULL,
activation VARCHAR(255) NOT NULL,
architecture VARCHAR(255) NOT NULL,
training_loss VARCHAR(3000), 
training_accuracy VARCHAR(3000), 
training_mae VARCHAR(3000), 
training_mse VARCHAR(3000), 
test_loss VARCHAR(255) NOT NULL, 
test_accuracy VARCHAR(255) NOT NULL, 
test_mae VARCHAR(255) NOT NULL, 
test_mse VARCHAR(255) NOT NULL,
label VARCHAR(255),
keywords VARCHAR(255),
PRIMARY KEY (optimizer, activation, architecture, label, keywords) 
);