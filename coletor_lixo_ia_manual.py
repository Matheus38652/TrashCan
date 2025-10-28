import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
import numpy as np
import os
import json
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import MobileNetV2
from PIL import Image, ImageTk

model = None
class_indices = None
cap = None
label_img = None

def verificar_estrutura_pasta(pasta_imagens):
    if not os.path.exists(pasta_imagens): return False
    subpastas = [f.path for f in os.scandir(pasta_imagens) if f.is_dir()]
    if len(subpastas) == 0: return False
    for subpasta in subpastas:
        if len([f for f in os.scandir(subpasta) if f.is_file()]) == 0: return False
    return True

def criar_modelo_com_transfer_learning(numero_de_classes):
    base_model = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
    base_model.trainable = False
    
    x = base_model.output
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    
    predictions = Dense(numero_de_classes, activation='softmax')(x) 
    
    model = Model(inputs=base_model.input, outputs=predictions)
    return model

def treinar_modelo(pasta_imagens):
    global model, class_indices
    if not verificar_estrutura_pasta(pasta_imagens):
        messagebox.showerror("Erro de Estrutura", "A pasta selecionada não contém subpastas com imagens. Verifique a estrutura.")
        return
    
    print("Iniciando o treinamento do modelo...")
    messagebox.showinfo("Treinamento", "O treinamento foi iniciado. Por favor, aguarde a mensagem de conclusão.")

    datagen = ImageDataGenerator(rescale=1./255, rotation_range=40, width_shift_range=0.2, height_shift_range=0.2, shear_range=0.2, zoom_range=0.2, horizontal_flip=True, fill_mode='nearest')
    train_generator = datagen.flow_from_directory(pasta_imagens, target_size=(224, 224), batch_size=32, class_mode='categorical')

    model = criar_modelo_com_transfer_learning(len(train_generator.class_indices))
    
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(train_generator, epochs=20)
    
    # Salva o modelo e classes com nomes específicos para o projeto de lixo
    model.save('modelo_lixo.h5')
    with open('classes_lixo.json', 'w') as f:
        json.dump(train_generator.class_indices, f)

    acc_final = history.history['accuracy'][-1] * 100
    messagebox.showinfo("Treinamento Concluído", f"Modelo de lixo treinado com precisão final de {acc_final:.2f}% e salvo com sucesso!")
    
    # Após treinar, carrega o modelo para uso imediato
    carregar_modelo()


def carregar_modelo():
    global model, class_indices, btn_analisar_snapshot # Adiciona referência ao novo botão
    try:
        model = load_model('modelo_lixo.h5')
        with open('classes_lixo.json', 'r') as f:
            class_indices = json.load(f)
        
        # Habilita o botão de analisar assim que o modelo é carregado
        if btn_analisar_snapshot:
            btn_analisar_snapshot.config(state=tk.NORMAL)
        print("Modelo de lixo carregado com sucesso.")
        return True
    except Exception as e:
        print(f"Erro ao carregar o modelo 'modelo_lixo.h5': {e}")
        return False

def reconhecer_objeto(frame, class_indices):
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image).resize((224, 224))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    
    previsao = model.predict(image)
    classe_index = np.argmax(previsao)
    confianca = np.max(previsao) * 100
    
    indices_para_classes = {v: k for k, v in class_indices.items()}
    nome_classe = indices_para_classes[classe_index]
    
    return nome_classe, confianca

# --- LÓGICA DA CÂMERA MODIFICADA ---

def mostrar_feed_camera():
    """Apenas exibe o feed da câmera, sem fazer análise contínua."""
    global cap, label_img
    if cap and cap.isOpened():
        ret, frame = cap.read()
        if ret:
            # Converte a imagem para exibição no Tkinter
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb)
            img_tk = ImageTk.PhotoImage(img_pil)
            label_img.config(image=img_tk)
            label_img.image = img_tk
        
        # Chama a si mesmo para criar o efeito de vídeo
        label_img.after(10, mostrar_feed_camera)

def ativar_camera():
    """Liga a câmera e inicia o feed de vídeo."""
    global cap, label_img
    if cap is None or not cap.isOpened():
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            messagebox.showerror("Erro de Câmera", "Não foi possível acessar a câmera.")
            cap = None
            return
        
        # Inicia o loop que apenas MOSTRA a imagem
        mostrar_feed_camera()
        
        # Habilita o botão de análise se o modelo já estiver carregado
        if model:
             btn_analisar_snapshot.config(state=tk.NORMAL)

def analisar_snapshot():
    """Tira uma ÚNICA foto e a analisa."""
    global cap, model, class_indices
    
    if not cap or not cap.isOpened():
        messagebox.showwarning("Aviso", "A câmera não está ativa. Por favor, inicie a câmera primeiro.")
        return
        
    if not model:
        messagebox.showwarning("Aviso", "O modelo não está carregado. Por favor, treine ou carregue um modelo.")
        return

    print("\n--- Analisando Snapshot ---")
    ret, frame = cap.read()
    if ret:
        # Chama a IA para analisar o quadro capturado
        classe, confianca = reconhecer_objeto(frame, class_indices)
        
        # *** ESTE É O "GATILHO" PARA O ROBÔ ***
        # No Raspberry Pi, aqui você enviará o comando para o Arduino
        print(f"Objeto Detectado: {classe}")
        print(f"ConfianCA: {confianca:.2f}%")
        
        # Define um limiar de confiança
        if classe == "bola_de_papel" and confianca > 75:
            print("GATILHO: Bola de papel encontrada! Acionando coleta...")
            # Ex: enviar_comando_arduino("COLETAR")
            texto_resultado = f"BOLA DE PAPEL! ({confianca:.1f}%)"
            cor = (0, 255, 0) # Verde
        else:
            print("GATILHO: Nenhum lixo alvo encontrado.")
            # Ex: enviar_comando_arduino("GIRAR")
            texto_resultado = f"{classe.capitalize()} ({confianca:.1f}%)"
            cor = (0, 0, 255) # Vermelho

        # Exibe o resultado na imagem por um momento (precisamos atualizar o frame)
        cv2.putText(frame, texto_resultado, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, cor, 2)
        
        # Atualiza a imagem no Tkinter para mostrar o resultado
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        img_tk = ImageTk.PhotoImage(img_pil)
        label_img.config(image=img_tk)
        label_img.image = img_tk
        
    else:
        print("Erro ao capturar imagem da câmera.")


def selecionar_pasta_e_treinar():
    pasta_imagens = filedialog.askdirectory()
    if pasta_imagens:
        treinar_modelo(pasta_imagens)


def interface_grafica():
    global label_img, btn_analisar_snapshot
    root = tk.Tk()
    root.title("Cérebro do Coletor de Lixo") # Título atualizado
    root.geometry("700x600")

    btn_treinar = tk.Button(root, text="Treinar Modelo", command=selecionar_pasta_e_treinar)
    btn_treinar.pack(pady=10)
    
    label_img = tk.Label(root)
    label_img.pack(pady=10)

    btn_camera = tk.Button(root, text="Iniciar Câmera", command=ativar_camera)
    btn_camera.pack(pady=5)

    # Novo botão para análise de snapshot
    btn_analisar_snapshot = tk.Button(root, text="Analisar Imagem (Snapshot)", state=tk.DISABLED, command=analisar_snapshot)
    btn_analisar_snapshot.pack(pady=5)
    
    # Tenta carregar o modelo ao iniciar
    carregar_modelo() 

    root.mainloop()

    if cap:
        cap.release()

if __name__ == "__main__":
    interface_grafica()