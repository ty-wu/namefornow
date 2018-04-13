import numpy as np
import sys

'''DO NOT modify this'''
# Given coefficients w,
# generate noisy random data points X, Y
# Where Y is approximately (X transpose)*w
# This is used to test gradient_descent
def generate_data(npoints, w):
	dim = np.size(w)
	X = np.random.uniform(-10, 10, (dim, npoints))
	Y = np.dot(X.T, w)
	err = np.random.normal(0, 1, (npoints))
	return X, Y+err

'''TODO'''
# Returns the exact best fit w to the data
def exact_solution(X, Y):
	# Implement this function
	# Can be done in one line
	# Use numpy matrix functions, like
		# np.linalg.inv for inverses
		# np.dot(X, Y) for matrix multiplication
		# X.T to take the transpose of X
		# np.eye to get identity matrices
		# Search google for more documentation
		# Make sure you understand what the functions do
		# If you haven't had linear algebra before, this
		# is your chance to learn some of the basics
		

	w = np.dot(np.dot(np.linalg.inv(np.add(np.eye(np.shape(X)[0]), np.dot(X, X.T))), X), Y)
	return w

'''TODO'''
# Starts with an initial guess for w
# and performs gradient descent until it converges

# Function to calculate the gradience of the loss function 
def calcGrad(X, Y, w):
	return np.substract(w, np.dot(X, np.subtract(Y, np.dot(X.T, w))))

# apply gradient descent
def gradient_descent(X, Y, eta=0.00001):
	w = np.zeros(X.shape[0])
	grad = calcGrad(X, Y, w)

	while(grad >= 0.001):
		w = np.subtract(w, eta*grad)
		grad = calculate(X, Y, w)

	return w
	# Implement gradient descent here



	return w


if __name__ == '__main__':
	if len(sys.argv) == 1:
		# Test if gradient descent was implemented properly
		w = np.random.normal(0,1,(10))
		X, Y = generate_data(100, w)
		w_exact = exact_solution(X, Y)
		w_solved = gradient_descent(X, Y)
		print w_exact
		print w_solved
		if np.max(abs(w_exact-w_solved)) < 0.01:
			print "Gradient descent working!"
		else:
			print "Gradient descent not working"

	else:
		# Extract the online_shares dataset
		fp = open(sys.argv[1], 'r')
		lines = fp.readlines()
		features = [st.strip() for st in lines[0].split(',')]
		features.pop() # Get rid of shares label
		data = np.genfromtxt(sys.argv[1], delimiter=',')

		X = data[1:, :data.shape[1]-1].T
		Y = data[1:, data.shape[1]-1]

		xmeans = np.mean(X, axis=0)
		ymean = np.mean(Y)
		xstdevs = np.std(X,axis=0)
		ystdev = np.std(Y)

		# Normalize the data for numerical stability
		X_normalized = (X-xmeans)/xstdevs
		Y_normalized = (Y-ymean)/ystdev

		w_exact = exact_solution(X_normalized, Y_normalized)

		y_pred = ystdev*np.dot(X_normalized.T, w_exact)+ymean

		rmse = np.linalg.norm(Y-y_pred)/np.sqrt(len(Y))
		print "Root mean squared error = ", rmse
		print "Standard deviation", np.std(Y)

		# Sort descending by absolute value of weights
		sorted_inds = np.argsort(abs(w_exact))[::-1]

		# Print out weights
		for i in range(len(w_exact)):
			print w_exact[sorted_inds[i]], ',', features[sorted_inds[i]]


