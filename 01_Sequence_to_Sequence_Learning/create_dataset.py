import csv
import random
from faker import Faker  # pip install faker

# ========== CONFIGURACIÓN ==========
NUM_EXAMPLES = 10000   # o 20000
OUTPUT_FILE = "dataset_10000.csv"
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
fake = Faker('en_US')   # Nombres en español (puedes cambiar a 'en_US')

# ========== COMPONENTES (ampliados) ==========

# Personas (nombres realistas)
people = [
    "Sarah", "Alex", "Mark", "Emily", "David", "Laura", "Carlos", "Marta",
    "Juan", "Ana", "Luis", "Elena", "Pablo", "Sofia", "Diego", "Claudia",
    "el equipo de DevOps", "el cliente", "el gerente", "el jefe", "la secretaria",
    "el departamento de TI", "el equipo de marketing", "el proveedor"
]

# Acciones (muchas más)
actions = [
    "run the database backup",
    "deploy the container",
    "update the repository",
    "review the pull request",
    "finish the Python script",
    "fix the UI bug",
    "water the plants",
    "take out the trash",
    "clean the bathroom",
    "wash the car",
    "feed the dog",
    "pick up dry cleaning",
    "buy {item}",
    "Schedule a meeting with {person}",
    "send the report to {person}",
    "call {person}",
    "write the documentation",
    "test the new feature",
    "prepare the presentation",
    "organize the files",
    "respond to emails",
    "update the website",
    "check the server logs",
    "install the updates",
    "configure the firewall",
    "create user accounts",
    "generate the invoices",
    "order supplies",
    "clean the desk",
    "water the garden",
    "walk the dog",
    "buy groceries",
    "pay the bills",
    "schedule the maintenance",
    "book the travel",
    "confirm the reservation",
    "print the documents",
    "scan the receipts",
    "backup the photos",
    "charge the devices",
    "empty the recycle bin",
    "restock the fridge",
    "sweep the floor",
    "mop the kitchen",
    "do the laundry",
    "iron the clothes",
    "change the light bulb",
    "fix the leaky faucet"
]

# Elementos para comprar
items = [
    "protein powder", "HDMI cable", "a new keyboard", "milk", "apples", "bread",
    "coffee beans", "eggs", "orange juice", "butter", "cheese", "tomatoes",
    "lettuce", "chicken", "pasta", "rice", "toilet paper", "shampoo",
    "soap", "batteries", "light bulbs", "printer ink", "paper towels"
]

# Tiempos y plazos (más variados)
times = [
    "tonight", "tomorrow", "this weekend", "on Tuesday", "at midnight",
    "before 5 PM", "by Friday", "this afternoon", "in the morning",
    "in the evening", "at noon", "right away", "as soon as possible",
    "before lunch", "after dinner", "at 3 PM", "on Monday morning",
    "next week", "within an hour", "later today"
]

deadlines = ["Friday", "Tuesday", "Monday", "Wednesday", "Thursday", "next Monday", "by the end of the week"]

# Razones / contexto (para enriquecer el input, no aparecen en target)
reasons = [
    "because the boss is coming",
    "since we have a deadline",
    "it's urgent",
    "no pressure, but",
    "if you have a moment",
    "when you get a chance",
    "I know you're busy, but",
    "it's not a priority, but",
    "this is critical",
    "please don't forget",
    "I'll remind you later",
    "just a heads up",
    "friendly reminder",
    "as we discussed",
    "per the client's request",
    "to avoid issues",
    "to be safe",
    "we need to",
    "the system requires",
    "the team needs"
]

# Plantillas de introducción para 2 tareas
intro_templates_2 = [
    "Remind me to {action1} and I also have to {action2} {time}.",
    "I need to {action1} {time} and also {action2}.",
    "Don't forget to {action1} and {action2} {time}.",
    "Please remember to {action1} and {action2}.",
    "Could you {action1} and then {action2}?",
    "We have to {action1} and {action2} {time}.",
    "Let's {action1} and {action2} {time}.",
    "I have to {action1} and {action2} {time}.",
    "We need to {action1} and also {action2}.",
    "It's important to {action1} and {action2}.",
]

# Plantillas para 3 tareas (incluyen reunión)
intro_templates_3 = [
    "I have to {action1}, {action2}, and also meet with {person} {time}.",
    "We need to {action1}, {action2}, and then meet {person}.",
    "My tasks: {action1}, {action2}, and meet with {person}.",
    "I should {action1}, {action2}, and if possible meet {person}.",
    "Gotta {action1}, {action2}, and meet {person} {time}.",
    "Please {action1}, {action2}, and schedule a meeting with {person}.",
]

# ========== FUNCIONES DE GENERACIÓN ==========

def generate_single_action():
    """Genera una acción con posibles placeholders."""
    action = random.choice(actions)
    if '{item}' in action:
        action = action.format(item=random.choice(items))
    elif '{person}' in action:
        action = action.format(person=random.choice(people))
    return action

def generate_two_tasks():
    """Genera una entrada con exactamente 2 tareas."""
    a1 = generate_single_action()
    a2 = generate_single_action()
    while a2 == a1:
        a2 = generate_single_action()
    
    # Añadir tiempo a una de ellas (50% de probabilidad)
    if random.random() > 0.5:
        time = random.choice(times)
        if random.random() > 0.5:
            a1 = f"{a1} {time}"
        else:
            a2 = f"{a2} {time}"
    else:
        time = ""
    
    template = random.choice(intro_templates_2)
    # Enriquecer con contexto (50% de probabilidad)
    if random.random() > 0.5:
        reason = random.choice(reasons)
        if random.random() > 0.5:
            input_text = f"{reason}, " + template.format(action1=a1, action2=a2, time=time)
        else:
            input_text = template.format(action1=a1, action2=a2, time=time) + f", {reason}"
    else:
        input_text = template.format(action1=a1, action2=a2, time=time)
    
    target = f"1. {a1.capitalize()} | 2. {a2}"
    return input_text, target

def generate_three_tasks():
    """Genera una entrada con 3 tareas (incluye reunión con alguien)."""
    a1 = generate_single_action()
    a2 = generate_single_action()
    while a2 == a1:
        a2 = generate_single_action()
    person = random.choice(people)
    time = random.choice(times) if random.random() > 0.5 else ""
    
    template = random.choice(intro_templates_3)
    if time:
        input_text = template.format(action1=a1, action2=a2, person=person, time=time)
    else:
        input_text = template.format(action1=a1, action2=a2, person=person, time="")
    if random.random() > 0.6:
        reason = random.choice(reasons)
        input_text = f"{reason}, " + input_text
    
    target = f"1. {a1.capitalize()} | 2. {a2} | 3. Meet with {person}"
    return input_text, target

def generate_example():
    """Genera un ejemplo completo (2 o 3 tareas)."""
    if random.random() < 0.3:  # 30% de probabilidad de 3 tareas
        return generate_three_tasks()
    else:
        return generate_two_tasks()

# ========== GENERAR DATASET ==========

print(f"Generando {NUM_EXAMPLES} ejemplos realistas...")
data = []
for _ in range(NUM_EXAMPLES):
    inp, tgt = generate_example()
    data.append({"input_text": inp, "target_text": tgt})

# Guardar CSV
with open(OUTPUT_FILE, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=["input_text", "target_text"])
    writer.writeheader()
    writer.writerows(data)

print(f"✅ Dataset generado en '{OUTPUT_FILE}' con {len(data)} ejemplos.")