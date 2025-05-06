# Texturation

By **Jules Imbert** & **Clémence Hamon**, *underwater2025*.

## Ce que contient ce repos

Ce repos contient le code permettant de texturer un mesh 3D déjà réalisé. Pour fonctionner, il a besoin de prendre en entrée :
- L'ensemble des images downsampled droites et gauches.
- Le mesh 3D généré à partir du nuage de point calculé (peut être généré à la main avec `algorithme/ply/mesh_creation.py` en ajustant les paramètres à la main jusqu'à obtenir une surface satisfaisante).
- Les vecteurs de transformation qui donnent les positions de la caméra gauche pour chaque image par rapport à l'origine du repère.
- Les vecteurs de transformation qui donnent les positions de la caméra droite par rapport à la caméra gauche.
- Les paramètres de calibration ayant généré le nuage de points.

A la fin du pipeline complet, ce code permettra de produire la texture map (`.obj` et `.png`) qui permettra de visualiser le mesh texturé sans couture, dans un logiciel comme blender. 

## Le pipeline

Les programmes à exécuter sont, dans l'ordre :
- `FAST_build_Mpj_Wpj.py` pour construire les matrices de vue et de coût individuel (`Mpj_cam` et `Wpj_cam`) tout en nettoyant le mesh initial.
- `FAST_build_Wpqjk.py` pour construire le tenseur de coût croisé (`Wpqjk_cam`). Dans la version la plus récente du code, `Wpqjk_cam` est stocké sur le disque.
- `alpha_expansion.py` pour calculer le coloriage optimal (`M_final`) du mesh, qui minimise les coûts (individuel et croisé). Dans la version la plus récente, `Wpqjk_cam` est chargé en RAM depuis le disque.
- `seamless_two_cam.py` pour construire la texture map avec ce coloriage et supprimer les dernières coutures.

Ensuite, cette texture map est à importer dans blender pour visualiser le résultat. 

## Points importants à noter

Le principal problème de ce pipeline est la gestion de la mémoire. Ce problème prend plusieurs formes :

- La gestion mémoire obscure de python et de ses bibliothèques (`Open3d`, `Multiprocessing`, etc). Des crash arrivent fréquemment si les calculs sont trop gourmands (en particulier sur de gros meshs) et en parallèle. C'est la raison pour laquelle `main.py` ne fonctionne pas : python n'arrive pas à libérer la mémoire et une erreur de segmentation se produit. C'est pourquoi il convient d'exécuter les différents programmes les uns après les autres. 

- La taille des objets. Ce problème n'est pas propre à python, mais à la taille des objets que nous manipulons. $W_{p,q,j,k}$ principalement, qui est de taille considérable (en $\mathcal{O}(\frac{3}{2}K\times 4N^2)$ avec $K\gtrsim 10^6$ pour le caillou de bonne qualité, et $N = 52$). Ce tenseur peut être trop gros pour la RAM, et python plantera alors avec le message d'erreur `Killed`. Actuellement, lors de sa construction, il est stocké en direct dans le disque, puis ouvert en entier lors de son utilisation (c'est possible puisque moins d'objets sont alors chargés). Alpha-expansion l'utilise trop souvent pour pouvoir le stocker ailleurs qu'en RAM.   

Si un programme indiviuel plante également (soit en s'éteignant, soit avec un message d'erreur comme une segfault), commencez par diminuer le nombre de coeurs qui travaillent en même temps. Si ça ne fonctionne toujours pas, séparez le programme en plusieurs étapes : après le calcul d'un gros objet numpy, vous l'enregistrez au format `.npy`, puis la fois d'après repartez de cet objet là. 

Si le problème est que la mémoire est insuffisante, il faudra construre l'objet dans le disque et pas en RAM. Utilisez la bibliothèque `shelve`. comme dans la dernière version de `FAST_build_Wpqjk.py`.


__Mes recommandations pour éviter définitivement ce genre de déconvenues :__

- Reprendre ce pipeline en c++ sans passer par python, et gérer correctement la mémoire pour éviter l'actuelle segfault inexpliquable du `main`. (Notez que `OLD/` contient des versions du code non-optimisées pour python. Il peut être préférable de prendre ces fonctions comme point de départ pour ensuite l'accélerer en c++).

- En pré-traitement, découper le mesh et les images en plusieurs sous ensembles qui seront texturés individuellement. Les coutures entre les morceaux seront sans doute visibles, mais le code de *seamless* pourra être adapté pour remédier à cela. 

## Variables et notations utilisées dans le code

La plupart des parties de ce code utilisent les notations suivantes :

### Images
- `N` : Nombre de vues (i.e. d'images prises par le rig. Ainsi si $N = 52$, il, y a 52 images gauches et 52 images droites).
- `j`, `k` : Indices de vues 
- `h`, `w` : Dimensions de l'image $(h, w)$.
- `Vjyxc_cam` : Tenseur image de shape $(N, h, w, 6)$. Contient la couleur par vue par pixel. Dans les anciennes versions du code, `Vjyxc` contenait juste les données des images gauches. Actuellement, le 6 est pour RGBRGB ou le premier RGB est pour l'image gauche, le second pour la droite.
- `Y` : Coordonnees $(x, y) \in \mathbb{R}^2$ d'un pixel.

### Mesh
- `K` : Nombre de faces du mesh.
- `p`, `q` : Indices de pixels du mesh (on parlera indistinctement de *pixels*, de *triangles* ou de *faces*).
- `(p, q)` : arete du mesh entre la face `p` et la face `q`.
- `v1`, `v2` : Indices de sommets du mesh (de *vertices*).
- `n_p`: Vecteur normal $\vec{n}_p$ à la face p.
- `n_l`ou `n_r` : Vecteur normal à la caméra gauche ou droite dans le repère monde.
- `X` : coordonnees $(x, y, z) \in \mathbb{R}^3$ d'un point du mesh.
- `edges_set` :   dict au format `{(p, q) : (v1, v2)}` qui associe les faces de l'edge $(p,q)$ aux indices des deux sommets qui forment l'edge. Seules les aretes ou $p < q$ sont présentes dans cette structure.

### Tenseurs

- `Mpj_cam` : Tenseur de visibilité de shape $(K, N, 2)$ contenant des booléens. $$M_{p,j,\epsilon} = \begin{cases} \text{True si la face p est visible sur la vue j de la caméra }\epsilon \\ \text{False sinon}\end{cases}$$ où $\epsilon$ vaut $0$ pour la caméra gauche et $1$ pour la caméra droite. 

- `Wpj_cam` : Tenseur de coût individuel de shape $(K, N, 2)$. $$W_{p,j,\epsilon} = \begin{cases} +\infty \text{ si la face p n'est pas visible depuis la vue j} \\ w_{p,j} \in \mathbb{R}_+^* \text{ le coût de la face avec la vue sinon}\end{cases}$$

- `Wpqjk_cam` : Dict au format `{(p, q) : {(j, k) : float}}`. Contient les valeurs non nulles et non infinies du coût croisé, où $p < q$. Cette fois-ci, $j, k \in [0, 2N[$, où $[0, N[$ est pour les images gauches et $[N, 2N[$ pour les droites.

- `full_Wpqjk` : Fonction. Contient toutes les valeurs du coût croisé. 

$$W_{p,q,j,k} = \begin{cases} +\infty \text{ si p n'est pas visible depuis j ou q n'est pas visible depuis k} \\ 0 \text{ si } j = k \\ \int_{E_{p,q}}d_{RGB}(V_{j,y,x,:}, V_{k,y,x,:})dX \text{ le coût croisé sinon}\end{cases}$$ 

### Alpha expansion
- `M` : vecteur de labels de taille $K$. $M_p \in \{0, ..., 2N\}$ est la vue qui colorie la face $p$.
- `alpha` : la vue que l'on cherche a étendre par un alpha-move

## Hyperparamètres

- $\theta_{max}$ : angle maximale autorisé entre la normale à la caméra et la normale à la face du mesh, au dessus duquel on considère qu'une face n'est plus visible depuis une vue donnée. Son cosinus est implémenté sous le nom de `cos_theta_max`. 
- `N_INTEGRATION` : le pas d'intégration dans le calcul du coût croisé. Pour des raisons de performances on prendra $N_{\text{int}} \gtrsim 10$.
- $\lambda$ : L'importance relative du coût croisé par rapport au coût individuel. N'est pas encore implémenté, doit être rajouté dans le calcul coût croisé (une simple multiplication).
- $\lambda_{\text{seam}}$ : Dans la supression des dernières coutures, le poids qu'on donne aux coutures
- La fonction de coût individuel : actuellement, $w_{p,j} = \text{distance à la caméra} \times (1.1 +  \vec{n}_\text{caméra j}\cdot \vec{n}_{p} )^2$. L'idée est de favoriser les faces proches et bien en face de la caméra.

**Jules Imbert**
*Contact : jules.imbert@etu.minesparis.psl.eu* 
**Clémence Hamon**
*Contact : clemence.hamon@etu.minesparis.psl.eu*