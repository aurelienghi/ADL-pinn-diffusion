def mitchMor(x):
    dist = scipy.spatial.distance.pdist(x, 'euclidean')
    # In ascending order
    q = [1,2,5,10] #ignore 20,50
    phi_i = 0
    for mm in range(4):
        phi = np.sum(np.power(dist,-q[mm]))**(1/q[mm])
        if phi>phi_i:
            phi_i = phi

    return phi