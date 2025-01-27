import pygame
import random
import math
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

pygame.init()
pygame.mixer.init()

# Attempt to load sound effects; if not found, set them to None
shoot_sound = None
hit_sound = None
zombie_death_sound = None
level_up_sound = None

if os.path.isfile("shoot.wav"):
    shoot_sound = pygame.mixer.Sound("shoot.wav")
if os.path.isfile("hit.wav"):
    hit_sound = pygame.mixer.Sound("hit.wav")
if os.path.isfile("zombie_death.wav"):
    zombie_death_sound = pygame.mixer.Sound("zombie_death.wav")
if os.path.isfile("level_up.wav"):
    level_up_sound = pygame.mixer.Sound("level_up.wav")

# User requested screen size: 1000 x 800
SCREEN_WIDTH, SCREEN_HEIGHT = 1200, 700
FPS = 60
WHITE, RED, GREEN, BLACK, CYAN, YELLOW = (255, 255, 255), (255, 0, 0), (0, 255, 0), (0, 0, 0), (0, 255, 255), (255, 255, 0)
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Zombie Hunter with Reinforcement Learning")
clock = pygame.time.Clock()
font = pygame.font.Font(None, 36)

ACTION_SPACE = 5
STATE_SIZE = 5
GAMMA = 0.99
LEARNING_RATE = 0.001
INITIAL_EPSILON = 1.0
EPSILON_MIN = 0.1
EPSILON_DECAY = 0.995
BATCH_SIZE = 64
MEMORY_SIZE = 10000

MIN_DISTANCE = 40
TOUCH_DISTANCE = 20
POISON_DISTANCE = 10
CENTER_X, CENTER_Y = SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2
CENTER_RADIUS = 200
STAY_RADIUS = 100
STAY_TIME = 3

# Levels: level 1=25 zombies, level 2=50 zombies, level 3=75 zombies, level 4=100 zombies, level 5=300 zombies
LEVELS = [25, 50, 75, 100, 300]

class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(STATE_SIZE, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, ACTION_SPACE)
        )
    def forward(self, state):
        return self.fc(state)

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    def add(self, experience):
        self.buffer.append(experience)
    def sample(self, batch_size):
        indices = random.sample(range(len(self.buffer)), batch_size)
        return [self.buffer[idx] for idx in indices]
    def __len__(self):
        return len(self.buffer)

class Zombie:
    def __init__(self, spawn_position):
        self.x, self.y = spawn_position
        self.health = 5
        self.speed = 1.5
        self.is_boss = False
    def move_toward(self, tx, ty):
        dx, dy = tx - self.x, ty - self.y
        dist = math.sqrt(dx*dx+dy*dy)
        if dist>0:
            dx, dy = dx/dist, dy/dist
            self.x+=dx*self.speed
            self.y+=dy*self.speed
        # Bounce off walls
        if self.x<0:
            self.x=0
        if self.x>SCREEN_WIDTH:
            self.x=SCREEN_WIDTH
        if self.y<0:
            self.y=0
        if self.y>SCREEN_HEIGHT:
            self.y=SCREEN_HEIGHT
    def draw(self):
        pygame.draw.circle(screen, RED, (int(self.x), int(self.y)), 8)
        pygame.draw.rect(screen, RED, (self.x-10, self.y-15, 20, 3))
        pygame.draw.rect(screen, GREEN, (self.x-10, self.y-15, 20*(self.health/5), 3))

class Bullet:
    def __init__(self,x,y,t):
        self.x=x
        self.y=y
        self.speed=10
        self.target=t
    def move(self):
        dx,dy=self.target.x-self.x,self.target.y-self.y
        dist=math.sqrt(dx*dx+dy*dy)
        if dist>0:
            dx,dy=dx/dist,dy/dist
            self.x+=dx*self.speed
            self.y+=dy*self.speed
    def draw(self):
        # Draw bullet as a small white circle to show bullet marks
        pygame.draw.circle(screen, WHITE, (int(self.x), int(self.y)),3)

def spawn_zombies(n):
    zombies=[]
    for _ in range(n):
        side = random.choice(['top','bottom','left','right'])
        if side=='top':
            sx=random.randint(0,SCREEN_WIDTH)
            sy=0
        elif side=='bottom':
            sx=random.randint(0,SCREEN_WIDTH)
            sy=SCREEN_HEIGHT
        elif side=='left':
            sx=0
            sy=random.randint(0,SCREEN_HEIGHT)
        else:
            sx=SCREEN_WIDTH
            sy=random.randint(0,SCREEN_HEIGHT)
        z=Zombie((sx,sy))
        zombies.append(z)
    return zombies

def resolve_overlaps(ai,zombies):
    for i in range(len(zombies)):
        for j in range(i+1,len(zombies)):
            dx=zombies[j].x - zombies[i].x
            dy=zombies[j].y - zombies[i].y
            dist=math.sqrt(dx*dx+dy*dy)
            if dist<MIN_DISTANCE and dist>0:
                ov=MIN_DISTANCE-dist
                px=(dx/dist)*(ov/2)
                py=(dy/dist)*(ov/2)
                zombies[i].x-=px
                zombies[i].y-=py
                zombies[j].x+=px
                zombies[j].y+=py
    d=[(zo,math.sqrt((zo.x - ai.x)**2+(zo.y - ai.y)**2)) for zo in zombies]
    d.sort(key=lambda x:x[1])
    for zt,dist in d[3:]:
        if dist<TOUCH_DISTANCE and dist>0:
            ov=TOUCH_DISTANCE-dist
            dx=zt.x - ai.x
            dy=zt.y - ai.y
            dd=math.sqrt(dx*dx+dy*dy)
            if dd>0:
                px=(dx/dd)*ov
                py=(dy/dd)*ov
                zt.x+=px
                zt.y+=py

class AISaviorRL:
    def __init__(self):
        self.x, self.y = SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2
        self.health = 100
        self.speed = 4
        self.bullets = []
        self.kills = 0
        self.rotation_angle = 0
        self.q_network = DQN().to(device)
        self.target_network = DQN().to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=LEARNING_RATE)
        self.replay_buffer = ReplayBuffer(MEMORY_SIZE)
        self.steps = 0
        self.highest_kills = 0
        self.trials = 0
        self.last_pos = (self.x, self.y)
        self.stay_timer = 0.0
        self.level = 1
        self.show_level_text = True
        self.level_display_timer = 0.0
        self.epsilon = INITIAL_EPSILON
        self.poison_timer = 0.0
        self.blink_timer = 0.0  # Add blink_timer


    def reset(self):
        self.x, self.y = CENTER_X, CENTER_Y  # Reset position to center
        self.health = 100
        self.bullets = []
        if self.kills > self.highest_kills:
            self.highest_kills = self.kills
        self.kills = 0
        self.trials += 1
        self.last_pos = (self.x, self.y)
        self.stay_timer = 0.0


    def get_state(self,z):
        if z:
            nz=min(z,key=lambda zz:math.sqrt((zz.x - self.x)**2+(zz.y-self.y)**2))
            zx,zy=nz.x,nz.y
        else:
            zx,zy=0,0
        return np.array([self.x/SCREEN_WIDTH,self.y/SCREEN_HEIGHT,zx/SCREEN_WIDTH,zy/SCREEN_HEIGHT,self.health/100],dtype=np.float32)

    def select_action(self,state):
        if random.random()<self.epsilon:
            return random.randint(0,ACTION_SPACE-1)
        s=torch.FloatTensor(state).unsqueeze(0).to(device)
        q=self.q_network(s)
        return torch.argmax(q).item()

    def take_action(self,action,zo):
        if action==0:
            self.y-=self.speed
        elif action==1:
            self.y+=self.speed
        elif action==2:
            self.x-=self.speed
        elif action==3:
            self.x+=self.speed
        elif action==4 and zo:
            nz=min(zo,key=lambda zz:math.sqrt((zz.x - self.x)**2+(zz.y-self.y)**2))
            self.bullets.append(Bullet(self.x,self.y,nz))
            if shoot_sound:
                shoot_sound.play()

        if self.x<0:
            self.x=0
        if self.x>SCREEN_WIDTH:
            self.x=SCREEN_WIDTH
        if self.y<0:
            self.y=0
        if self.y>SCREEN_HEIGHT:
            self.y=SCREEN_HEIGHT

        moved_dist=math.sqrt((self.x - self.last_pos[0])**2+(self.y - self.last_pos[1])**2)
        if moved_dist<=STAY_RADIUS:
            self.stay_timer+=1/FPS
            if self.stay_timer>=STAY_TIME:
                self.health-=5
                self.stay_timer=0.0
                if hit_sound:
                    hit_sound.play()
        else:
            self.last_pos=(self.x,self.y)
            self.stay_timer=0.0

    def update_damage(self, z):
        d = [(zo, math.sqrt((zo.x - self.x)**2 + (zo.y - self.y)**2)) for zo in z]
        d.sort(key=lambda x: x[1])
        touching = [zo for zo, dist in d[:3] if dist < TOUCH_DISTANCE]
        poisoning = [zo for zo, dist in d[:3] if dist < POISON_DISTANCE]
    
        for zo in touching:
            self.health -= (10 / FPS)
            self.blink_timer = 1.0  # Start blinking animation
            if hit_sound:
                hit_sound.play()
    
        for zo in poisoning:
            self.health -= (10 / FPS)
            self.blink_timer = 1.0  # Start blinking animation
    
        if self.health < 0:
            self.health = 0



    def draw(self):
        self.rotation_angle += 5
        self.rotation_angle %= 360
        diamond_points = [
            (self.x, self.y - 10),
            (self.x - 10, self.y),
            (self.x, self.y + 10),
            (self.x + 10, self.y),
        ]
        rp = []
        for px, py in diamond_points:
            rx = self.x + math.cos(math.radians(self.rotation_angle)) * (px - self.x) - math.sin(math.radians(self.rotation_angle)) * (py - self.y)
            ry = self.y + math.sin(math.radians(self.rotation_angle)) * (px - self.x) + math.cos(math.radians(self.rotation_angle)) * (py - self.y)
            rp.append((rx, ry))
        pygame.draw.polygon(screen, CYAN, rp)
    
        # Draw bullets
        for b in self.bullets:
            b.draw()
    
        # Blinking circle animation
        if self.blink_timer > 0:
            alpha = int((math.sin(self.blink_timer * math.pi * 2) + 1) * 127.5)  # Blink effect
            overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
            pygame.draw.circle(overlay, (255, 255, 0, alpha), (int(self.x), int(self.y)), 50, 5)  # Yellow circle
            screen.blit(overlay, (0, 0))
            self.blink_timer -= 1 / FPS



    def train(self):
        if len(self.replay_buffer)<BATCH_SIZE:
            return
        batch=self.replay_buffer.sample(BATCH_SIZE)
        states,actions,rewards,next_states,dones=zip(*batch)
        states=torch.FloatTensor(states).to(device)
        actions=torch.LongTensor(actions).unsqueeze(1).to(device)
        rewards=torch.FloatTensor(rewards).to(device)
        next_states=torch.FloatTensor(next_states).to(device)
        dones=torch.FloatTensor(dones).to(device)
        qv=self.q_network(states).gather(1,actions)
        nqv=self.target_network(next_states).max(1)[0].detach()
        tg=rewards+(1-dones)*GAMMA*nqv
        loss=nn.MSELoss()(qv.squeeze(),tg)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        if self.steps%100==0:
            self.target_network.load_state_dict(self.q_network.state_dict())
            # Print RL info to terminal
            print(f"Step: {self.steps}, Loss: {loss.item():.4f}, EPSILON: {self.epsilon:.4f}")

        self.steps+=1

        if self.epsilon>EPSILON_MIN:
            self.epsilon*=EPSILON_DECAY

    def advance_level(self):
        self.level += 1
        self.show_level_text = True
        self.level_display_timer = 0.0
        self.x, self.y = CENTER_X, CENTER_Y  # Reset AI Savior position to center
        if level_up_sound:
            level_up_sound.play()


device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
ai_savior=AISaviorRL()

def start_level(level):
    count=LEVELS[level-1]
    z=spawn_zombies(count)
    return z

zombies = start_level(ai_savior.level)
running=True

while running:
    dt=clock.tick(FPS)/1000.0
    for event in pygame.event.get():
        if event.type==pygame.QUIT:
            running=False

    st=ai_savior.get_state(zombies)
    act=ai_savior.select_action(st)
    ai_savior.take_action(act,zombies)

    for zz in zombies:
        zz.move_toward(ai_savior.x,ai_savior.y)

    resolve_overlaps(ai_savior,zombies)
    ai_savior.update_damage(zombies)

    for bul in ai_savior.bullets[:]:
        bul.move()
        hit_any=False
        for zz in zombies[:]:
            dist=math.sqrt((zz.x - bul.x)**2+(zz.y - bul.y)**2)
            if dist<8:
                zz.health-=1
                ai_savior.bullets.remove(bul)
                hit_any=True
                if hit_sound:
                    hit_sound.play()
                if zz.health<=0:
                    zombies.remove(zz)
                    ai_savior.kills+=1
                    ai_savior.health+=5
                    if zombie_death_sound:
                        zombie_death_sound.play()
                break
        if hit_any:
            continue

    done=ai_savior.health<=0
    if done:
        ai_savior.reset()
        ai_savior.level=1
        zombies = start_level(ai_savior.level)

    if len(zombies)==0 and not done:
        if ai_savior.level<5:
            ai_savior.advance_level()
            zombies = start_level(ai_savior.level)
        else:
            ai_savior.advance_level()
            ai_savior.reset()
            ai_savior.level=1
            zombies = start_level(ai_savior.level)

    nst=ai_savior.get_state(zombies)
    rew=-1 if ai_savior.health>0 else -100
    distc=math.sqrt((ai_savior.x - CENTER_X)**2+(ai_savior.y - CENTER_Y)**2)
    if distc>CENTER_RADIUS:
        rew+=0.5

    if len(zombies)>0:
        nearest_zombie=min(zombies,key=lambda zz:math.sqrt((zz.x - ai_savior.x)**2+(zz.y - ai_savior.y)**2))
        nz_dist=math.sqrt((nearest_zombie.x - ai_savior.x)**2+(nearest_zombie.y - ai_savior.y)**2)
        if nz_dist>300:
            rew+=1.0
        elif nz_dist<100:
            rew-=1.0
    else:
        rew+=0.5

    ai_savior.replay_buffer.add((st,act,rew,nst,done))
    ai_savior.train()

    screen.fill(BLACK)
    for zz in zombies:
        zz.draw()
    ai_savior.draw()
    ht=font.render(f"Health: {ai_savior.health:.1f}",True,WHITE)
    kt=font.render(f"Kills: {ai_savior.kills}",True,WHITE)
    zt=font.render(f"Zombies Left: {len(zombies)}",True,WHITE)
    hkt=font.render(f"Highest Kills: {ai_savior.highest_kills}",True,WHITE)
    tt=font.render(f"Number of Trials: {ai_savior.trials}",True,WHITE)
    lvl_text=font.render(f"Level: {ai_savior.level}",True,WHITE)

    screen.blit(ht,(10,10))
    screen.blit(kt,(10,40))
    screen.blit(zt,(10,70))
    screen.blit(hkt,(SCREEN_WIDTH - 300,10))
    screen.blit(tt,(SCREEN_WIDTH - 300,40))
    screen.blit(lvl_text,(SCREEN_WIDTH//2 - 50,10))

    if ai_savior.show_level_text:
        level_msg=font.render(f"LEVEL {ai_savior.level}",True,WHITE)
        screen.blit(level_msg,(SCREEN_WIDTH//2 - 50, SCREEN_HEIGHT//2))
        ai_savior.level_display_timer+=dt
        if ai_savior.level_display_timer>2:
            ai_savior.show_level_text=False

    pygame.display.flip()

pygame.quit()
