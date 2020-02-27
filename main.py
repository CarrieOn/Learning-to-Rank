
############################## Load Dataset ###########################
print ("----------------------- Load Data ------------------")

print(">>> Import functions.py")
from functions import *

print(">>> Loading Dataset...")
x = get_data(r'Querylevelnorm_X.csv')
y = get_label(r'Querylevelnorm_t.csv')

print(">>> Spliting Dataset...")
x_train, x_test, x_val, y_train, y_test, y_val = split_data(x, y)
print("\n")


##################### Train Model on Train Dataset ####################
print ("--------------- Compute Hyper-parameters -----------")

# Tune hyper-parameters here
M = 10
Lambda = 0.9
learning_rate = 0.01
sigma_factor = 10 # bigger sigma to capture more sparse data
print("M\t\t=", M)
print("Lambda\t\t=", Lambda)
print("learning_rate\t=", learning_rate)
print("sigma_factor\t=", sigma_factor)

# run SGD via eopochs and batches, add early_stopping
batch_size = 120
epochs = 500
early_stopping = 1.0e-5
print("batch_size\t=", batch_size)
print("epochs\t\t=", epochs)
print("early_stopping\t=", early_stopping)
print("\n")

print(">>> Computing Mu via k-means clustering...")
max_iters = 200
Mu, memberships = k_means(x_train, M, max_iters)
print("    Take a look at one Mu:\n", Mu[0])

print(">>> Computing sigma_train...")
sigma = get_sigma(x_train, M, memberships) * sigma_factor 
print("    Sigma diagonal values for one cluster: \n", np.diagonal(sigma[0]))

print(">>> Computing phi_train...")
phi_train = get_phi_matrix(x_train, M, Mu, sigma)

print(">>> Computing phi_val...")
phi_val = get_phi_matrix(x_val, M, Mu, sigma)

print(">>> Computing phi_test...")
phi_test = get_phi_matrix(x_test, M, Mu, sigma)


########################### Closed-Form Solution ######################
print("\n")
print ("------ Closed Form with Radial Basis Function ------")

print(">>> Evaluating Erms (closed-form) on training dataset...")
w_closed_form_train = w_closed_form(y_train, phi_train, Lambda)
y_closed_form_train = np.dot(phi_train, w_closed_form_train)
err_closed_form_train = evaluate(y_closed_form_train, y_train)

print(">>> Evaluating Erms (closed-form) on validation dataset...")
y_closed_form_val = np.dot(phi_val, w_closed_form_train)
err_closed_form_val = evaluate(y_closed_form_val, y_val)

print(">>> Evaluating Erms (closed-form) on testing dataset...")
y_closed_form_test = np.dot(phi_test, w_closed_form_train)
err_closed_form_test = evaluate(y_closed_form_test, y_test)

print("\n")
print ("    E_rms Training\t= ", err_closed_form_train)
print ("    E_rms validation\t= ", err_closed_form_val)
print ("    E_rms Testing\t= ", err_closed_form_test)
print("\n")


############################# SGD Solution ############################
print ("----------- SGD with Radial Basis Function ---------")

print(">>> Evaluating Erms (SGD) on training dataset...")
w_sgd_train = w_sgd(x_train, y_train, M, phi_train, learning_rate, Lambda, batch_size, epochs, early_stopping)
y_sgd_train = np.dot(phi_train, w_sgd_train)
err_sgd_train = evaluate(y_sgd_train, y_train)

print(">>> Evaluating Erms (SGD) on validation dataset...")
y_sgd_val = np.dot(phi_val, w_sgd_train)
err_sgd_val = evaluate(y_sgd_val, y_val)

print(">>> Evaluating Erms (SGD) on testing dataset...")
y_sgd_test = np.dot(phi_test, w_sgd_train)
err_sgd_test = evaluate(y_sgd_test, y_test)

print("\n")
print ("    E_rms Training\t= ", err_sgd_train)
print ("    E_rms validation\t= ", err_sgd_val)
print ("    E_rms Testing\t= ", err_sgd_test)
print("\n")

print(">>> The END, thank you!")

