
def run_texture_generation():
    print("\n--- Début de la génération de la texture complète (R, G, B) ---")
    
    # Calcul des intensités pour chaque canal
    intensities_per_channel = []
    all_views_per_channel = []

    for channel_id in range(3):
        intensities = {view_id: intensity(view_id, channel_id) for view_id in range(n_views)}
        all_views = intensity_all_views(views, intensities, M)
        intensities_per_channel.append(intensities)
        all_views_per_channel.append(all_views)

    # Optimisation pour chaque canal
    optimal_x_per_channel = []

    for channel_id in range(3):
        g = build_g_function(all_views_per_channel[channel_id], L, M, index_map)
        x0 = np.zeros(n)
        res = least_squares(g, x0, jac='2-point')
        optimal_x_per_channel.append(res.x)
        print(f"Canal {channel_id} : Coût final = {res.x}")

    # Mise à jour des intensités optimisées
    for channel_id in range(3):
        all_views = all_views_per_channel[channel_id]
        optimal_x = optimal_x_per_channel[channel_id]
    
        for (i, j) in all_views:
            print(f"AVANT OPTIM  Vertex {i}, Vue {j} : Intensité = {all_views[(i, j)]}")

            idx = index_map.get((i, j), None)
            if idx is not None and (i, j) in M:
                all_views[(i, j)] += optimal_x[idx]
    
        for (i, j) in M:
            print(f"APRES OPTIM Vertex {i}, Vue {j} : Intensité optimisée = {all_views[(i, j)]}")

    # Génération de l'image de texture finale
    texture_size = 512
    texture_image = np.ones((texture_size, texture_size, 3), dtype=np.uint8) * 255

    for face_id, face in enumerate(mesh.faces):
        for view_id in best_views:
            h, w = texture_image.shape[:2]
            uvs_coord = uvs[face]
            img_coords = (uvs_coord * [w, h]).astype(int)
            img_coords = np.clip(img_coords, 0, [w - 1, h - 1])

            colors = []
            for vertex in face:
                rgb = [
                    all_views_per_channel[1].get((vertex, view_id), 0),  # vert
                    all_views_per_channel[2].get((vertex, view_id), 0)   # bleu
                ]
                if vertex == 1:
                    print(all_views_per_channel[1].get((vertex, view_id), 0))
                colors.append(rgb)

            texture_triangles(uvs_coord, colors, texture_image)

        # Affichage
        plt.figure(figsize=(8, 8))
        plt.imshow(texture_image, origin='lower')
        plt.axis('off')
        plt.title('Texture finale générée (R, G, B)')
        plt.show()

