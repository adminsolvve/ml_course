'''
    files=glob.glob(faces_folder + "s*/*.pgm")   # list of filenames   import glob
    img=np.array(PGM.open(faces_folder + "s" + str(person+1) + "/" + str(faces+1) + ".pgm"))
    print(img.shape)
    print(img)
    plt.imshow(img, cmap='hot')
    print(np.asarray(Xt).shape)
    plt.imshow(np.asarray(Xt)[10], cmap='hot')
'''

# not use
def read_pgm_file (file_name):
    """Return a raster of integers from a PGM as a list of lists """
    assert file_name.readline() == 'P5\n'
    (width, height) = [int(i) for i in file_name.readline().split()]
    depth = int(file_name.readline())
    assert depth <= 255
    raster = []
    for y in range(height):
        row = []
        for y in range(width):
            row.append(ord(file_name.read(1)))
        raster.append(row)
    return raster


%%time
fig = plt.figure(figsize=(faces_count_verif*X_verif_orig.shape[2]/32, person_count*X_verif_orig.shape[1]/32)) 
for i in range(X_verif_orig.shape[0]):
    fig.add_subplot(person_count, faces_count_verif, i+1)
    plt.imshow(X_verif_orig[i], cmap='hot')
plt.show()


