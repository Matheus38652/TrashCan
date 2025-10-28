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
btn_analise_auto = None
analise_automatica_ativa = False

TEMPO_DE_ESPERA_MS = 5000

# (Funções de treinar e carregar o modelo permanecem iguais...)
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
    
    model.save('modelo_lixo.h5')
    with open('classes_lixo.json', 'w') as f:
        json.dump(train_generator.class_indices, f)

    acc_final = history.history['accuracy'][-1] * 100
    messagebox.showinfo("Treinamento Concluído", f"Modelo de lixo treinado com precisão final de {acc_final:.2f}% e salvo com sucesso!")
    carregar_modelo()


def carregar_modelo():
    global model, class_indices, btn_analise_auto # <-- MUDANÇA
    try:
        model = load_model('modelo_lixo.h5')
        with open('classes_lixo.json', 'r') as f:
            class_indices = json.load(f)
        
        if btn_analise_auto:
            btn_analise_auto.config(state=tk.NORMAL)
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
    global cap, label_img
    if cap and cap.isOpened():
        ret, frame = cap.read()
        if ret:
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb)
            img_tk = ImageTk.PhotoImage(img_pil)
            label_img.config(image=img_tk)
            label_img.image = img_tk
        
        label_img.after(10, mostrar_feed_camera)

def ativar_camera():
    global cap, label_img, btn_analise_auto # <-- MUDANÇA
    if cap is None or not cap.isOpened():
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            messagebox.showerror("Erro de Câmera", "Não foi possível acessar a câmera.")
            cap = None
            return
        
        mostrar_feed_camera()
        
        # Habilita o botão de análise se o modelo já estiver carregado
        if model:
             btn_analise_auto.config(state=tk.NORMAL) # <-- MUDANÇA

# --- NOVA LÓGICA DE LOOP AUTOMÁTICO ---

def loop_de_analise():
    """Esta é a função principal do robô. Ela se auto-chama."""
    global analise_automatica_ativa, cap, model, class_indices
    
    # 1. Verifica se o "interruptor" está ligado
    if not analise_automatica_ativa:
        print("Análise automática parada.")
        return # Para a execução

    # 2. Executa a lógica de análise (o antigo analisar_snapshot)
    if not cap or not cap.isOpened() or not model:
        print("Câmera ou modelo não prontos. Parando loop.")
        alternar_analise_automatica() # Desliga o botão
        return

    print("\n--- Analisando Snapshot ---")
    ret, frame = cap.read()
    if ret:
        classe, confianca = reconhecer_objeto(frame, class_indices)
        
        print(f"Objeto Detectado: {classe}")
        print(f"ConfianCA: {confianca:.2f}%")
        
        if classe == "bola_de_papel" and confianca > 75:
            print("GATILHO: Bola de papel encontrada! Acionando coleta...")
            # Ex: enviar_comando_arduino("COLETAR")
            texto_resultado = f"BOLA DE PAPEL! ({confianca:.1f}%)"
            cor = (0, 255, 0)
        else:
            print("GATILHO: Nenhum lixo alvo encontrado.")
            # Ex: enviar_comando_arduino("GIRAR")
            texto_resultado = f"{classe.capitalize()} ({confianca:.1f}%)"
            cor = (0, 0, 255)

        # Atualiza a imagem no Tkinter para mostrar o resultado
        cv2.putText(frame, texto_resultado, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, cor, 2)
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        img_tk = ImageTk.PhotoImage(img_pil)
        label_img.config(image=img_tk)
        label_img.image = img_tk
        
    else:
        print("Erro ao capturar imagem da câmera.")

    # 3. Agenda a próxima execução
    # Chama a si mesma novamente após o tempo de espera definido
    label_img.after(TEMPO_DE_ESPERA_MS, loop_de_analise)

def alternar_analise_automatica():
    """Função "interruptor" para o botão Iniciar/Parar."""
    global analise_automatica_ativa, btn_analise_auto
    
    if analise_automatica_ativa:
        # Se estiver rodando, vamos parar
        analise_automatica_ativa = False
        btn_analise_auto.config(text="Iniciar Análise Automática", bg="SystemButtonFace")
    else:
        # Se estiver parado, vamos começar
        analise_automatica_ativa = True
        btn_analise_auto.config(text="Parar Análise (Executando...)", bg="lightcoral")
        
        # Inicia o loop pela primeira vez
        loop_de_analise()


def selecionar_pasta_e_treinar():
    pasta_imagens = filedialog.askdirectory()
    if pasta_imagens:
        treinar_modelo(pasta_imagens)

def interface_grafica():
    global label_img, btn_analise_auto # <-- MUDANÇA
    root = tk.Tk()
    root.title("Cérebro do Coletor de Lixo")
    root.geometry("700x600")

    btn_treinar = tk.Button(root, text="Treinar Modelo", command=selecionar_pasta_e_treinar)
    btn_treinar.pack(pady=10)
    
    label_img = tk.Label(root)
    label_img.pack(pady=10)

    btn_camera = tk.Button(root, text="Iniciar Câmera", command=ativar_camera)
    btn_camera.pack(pady=5)

    # Botão modificado para ligar/desligar o loop
    btn_analise_auto = tk.Button(root, text="Iniciar Análise Automática", state=tk.DISABLED, command=alternar_analise_automatica) # <-- MUDANÇA
    btn_analise_auto.pack(pady=5)
    
    carregar_modelo() 
    root.mainloop()

    if cap:
        cap.release()

if __name__ == "__main__":
    interface_grafica()