import numpy as np
import shap
from sklearn import model_selection
from keras import backend as K
from pywt import wavedec2, waverec2
from sklearn.utils import shuffle

# Train-test split
codes3, _, labels_vectors3, _ = model_selection.train_test_split(batch, labels_vecs, train_size=300, random_state=1)
codes3, labels_vectors3 = shuffle(codes3, labels_vectors3, random_state=1)

# Compute SHAP values for all images
shap_values_list = []
indexes_list = []

# Calculate SHAP values
def map2layer(x, layer):
    feed_dict = dict(zip([model.layers[0].input], [x]))
    return K.get_session().run(model.layers[layer].input, feed_dict)

for img in codes3:
    img_input = np.expand_dims(img, axis=0)
    e = shap.GradientExplainer(
        (model.layers[15].input, model.layers[-1].output),
        map2layer(img_input, 15),
        local_smoothing=0  # Std dev of smoothing noise
    )
    shap_values, indexes = e.shap_values(map2layer(img_input, 20), ranked_outputs=1)
    shap_values_list.append(shap_values)
    indexes_list.append(indexes)

# Get class names
class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
index_names_list = [np.vectorize(lambda x: class_names[int(x)])(indexes) for indexes in indexes_list]

# Stack SHAP values into a single array
shap_values_array = np.array(shap_values_list)

# Calculate mean SHAP values
mean_shap_values = np.mean(shap_values_array, axis=0)

# Select the most important k coefficients based on SHAP values
def select_top_k_shap_features(shap_values, k=1000):
    """
    选择 SHAP 值中最重要的前 k 个特征
    shap_values: [batch_size, height, width, channels]
    k: 选择的特征数量
    """
    shap_magnitude = np.abs(shap_values).sum(axis=(1, 2))  # 聚合 SHAP 值的影响
    top_k_indices = np.argsort(shap_magnitude.flatten())[-k:]  # 按值排序并选择前 k 个
    return top_k_indices

def apply_shap_selection_to_features(features, shap_values, method='top_k', k=10000):
    """
    根据 SHAP 值选择特征并返回四元数索引
    """
    if method == 'top_k':
        # 选择最重要的 k 个 SHAP 特征
        selected_indices = select_top_k_shap_features(shap_values, k)
    
    # 将 features 转换为 NumPy 数组，以便进行索引操作
    features = np.array(features)
    
    # 确保 selected_indices 不超过 features 的通道维度
    num_channels = features.shape[-1]
    selected_indices = np.clip(selected_indices, 0, num_channels - 1)
    
    # 确保每个四元数的所有分量（q_real, q_i, q_j, q_k）都被选中
    selected_features = []
    for i in range(0, len(selected_indices), 4):  # 每四个索引为一组
        selected_set = selected_indices[i:i + 4]
        
        # 如果没有四个分量，填充缺失的分量
        while len(selected_set) < 4:
            selected_set = np.concatenate((selected_set, [selected_set[-1]]))  # 填充
        
        selected_features.append(selected_set)
    
    complete_selected_features = []
    for i in range(0, len(selected_features), 4):
        complete_selected_features.append(selected_features[i:i + 4])  # 每 4 个通道为一组
    
    return complete_selected_features

def quaternion_wavelet_decomposition(features, wavelet='haar'):
    """
    对四元数特征进行小波分解
    features: 输入的四元数特征，形状为 (1, 25, 25, 64)
    """
    coeffs_list = []
    features = features[0]  # 提取批次维度中的特征
    
    # 对每个四元数（每四个通道作为一个四元数）进行小波分解
    for i in range(0, features.shape[2], 4):
        if i + 3 < features.shape[2]:  # 确保不会超出通道维度
            q_real = features[:, :, i]
            q_i = features[:, :, i + 1]
            q_j = features[:, :, i + 2]
            q_k = features[:, :, i + 3]
            
            # 对每个分量进行小波分解
            coeffs_real = wavedec2(q_real, wavelet)
            coeffs_i = wavedec2(q_i, wavelet)
            coeffs_j = wavedec2(q_j, wavelet)
            coeffs_k = wavedec2(q_k, wavelet)
            
            coeffs_list.append((coeffs_real, coeffs_i, coeffs_j, coeffs_k))
    
    return coeffs_list

def quaternion_wavelet_reconstruction(coeffs_list, wavelet='haar'):
    """
    基于四元数的小波系数进行重建
    coeffs_list: 四元数系数列表，每个元素是一个元组（实部系数、i系数、j系数、k系数）
    """
    reconstructed = []
    
    for coeffs in coeffs_list:
        if len(coeffs) != 4:  # 确保每个四元数有 4 个分量
            raise ValueError("每个四元数需要 4 个系数（实部、i、j、k）。")
        
        # 需要将每个分量的系数变成 (cA, (cH, cV, cD)) 形式
        # 对于每个分量，我们将其近似系数和细节系数合并为一个二元组
        coeffs_real = coeffs[0]  # 实部系数
        coeffs_i = coeffs[1]     # i 部分系数
        coeffs_j = coeffs[2]     # j 部分系数
        coeffs_k = coeffs[3]     # k 部分系数
        
        # 对每个系数分量执行重建
        q_real_rec = waverec2(coeffs_real, wavelet)
        q_i_rec = waverec2(coeffs_i, wavelet)
        q_j_rec = waverec2(coeffs_j, wavelet)
        q_k_rec = waverec2(coeffs_k, wavelet)
        
        # 裁剪到 (25, 25) 的尺寸（如有必要）
        q_real_rec = q_real_rec[:25, :25]
        q_i_rec = q_i_rec[:25, :25]
        q_j_rec = q_j_rec[:25, :25]
        q_k_rec = q_k_rec[:25, :25]
        
        # 合并四个分量，形成完整的四元数
        reconstructed.append([q_real_rec, q_i_rec, q_j_rec, q_k_rec])
    
    # 创建一个空数组来保存重建后的特征
    rec_array = np.zeros((1, 25, 25, 64))
    
    # 将重建后的值填充到 rec_array 中
    for i in range(len(reconstructed)):
        for j in range(4):  # 实部、i、j、k 四个分量
            if i * 4 + j < 64:  # 确保不会越界
                rec_array[0, :, :, i * 4 + j] = reconstructed[i][j]
    
    return rec_array


# 提取中间特征并进行小波分解
intermediate_features_list = []
for img in codes:
    img_input = np.expand_dims(img, axis=0)
    # 提取中间特征
    intermediate_features = K.get_session().run(model.layers[15].output, feed_dict={model.layers[0].input: img_input})
    intermediate_features_list.append(intermediate_features)

# 对所有中间特征执行四元数小波分解
all_coeffs_list = []
for features in intermediate_features_list:
    coeffs = quaternion_wavelet_decomposition(features)
    all_coeffs_list.append(coeffs)

# 使用选择的系数进行特征重建
#selected_reconstructed_images = []
#for selected_coeffs in selected_coeffs_list:
#    rec_img = quaternion_wavelet_reconstruction(selected_coeffs)
#    selected_reconstructed_images.append(rec_img)
    
# 使用所有的系数进行特征重建
selected_reconstructed_images = []
for selected_coeffs in all_coeffs_list:
    rec_img = quaternion_wavelet_reconstruction(selected_coeffs)
    selected_reconstructed_images.append(rec_img)


# 转换为 NumPy 数组
#rec_images = np.array(reconstructed_images)
selected_rec_images = np.array(selected_reconstructed_images)

# 重新调整形状
selected_rec_images_reshaped = selected_rec_images.reshape(13671, 25, 25, 64)








