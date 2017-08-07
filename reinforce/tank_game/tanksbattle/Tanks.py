#!/usr/bin/python
# coding=utf-8
import os

import cv2
import numpy as np 
import os, pygame, time, random, uuid, sys

import traceback

sprites = None
screen = None
players = []
enemies = []
bullets = []
bonuses = []
labels = []
castle = None
gtimer = None

GAME_LEVEL = 2

class myRect(pygame.Rect):
    """ Add type property """
    def __init__(self, left, top, width, height, type):
        pygame.Rect.__init__(self, left, top, width, height)
        self.type = type

class Timer(object):
    def __init__(self):
        self.timers = []

    def add(self, interval, f, repeat = -1):
        options = {
            "interval"  : interval,
            "callback"  : f,
            "repeat"        : repeat,
            "times"         : 0,
            "time"          : 0,
            "uuid"          : uuid.uuid4()
        }
        self.timers.append(options)

        return options["uuid"]

    def destroy(self, uuid_nr):
        for timer in self.timers:
            if timer["uuid"] == uuid_nr:
                self.timers.remove(timer)
                return

    def update(self, time_passed):
        for timer in self.timers:
            timer["time"] += time_passed
            if timer["time"] > timer["interval"]:
                timer["time"] -= timer["interval"]
                timer["times"] += 1
                if timer["repeat"] > -1 and timer["times"] == timer["repeat"]:
                    self.timers.remove(timer)
                try:
                    timer["callback"]()
                except:
                    try:
                        self.timers.remove(timer)
                    except:
                        pass

class Castle():
    """ Player's castle/fortress """

    (STATE_STANDING, STATE_DESTROYED, STATE_EXPLODING) = range(3)

    def __init__(self):

        global sprites

        # images
        self.img_undamaged = sprites.subsurface(0, 15*2, 16*2, 16*2)
        self.img_destroyed = sprites.subsurface(16*2, 15*2, 16*2, 16*2)

        # init position
        self.rect = pygame.Rect(12*16, 24*16, 32, 32)

        # start w/ undamaged and shiny castle
        self.rebuild()

    def draw(self):
        """ Draw castle """
        global screen

        screen.blit(self.image, self.rect.topleft)

        if self.state == self.STATE_EXPLODING:
            if not self.explosion.active:
                self.state = self.STATE_DESTROYED
                del self.explosion
            else:
                self.explosion.draw()

    def rebuild(self):
        """ Reset castle """
        self.state = self.STATE_STANDING
        self.image = self.img_undamaged
        self.active = True

    def destroy(self):
        """ Destroy castle """
        self.state = self.STATE_EXPLODING
        self.explosion = Explosion(self.rect.topleft)
        self.image = self.img_destroyed
        self.active = False

class Bonus():
    """ Various power-ups
    When bonus is spawned, it begins flashing and after some time dissapears

    Available bonusses:
        grenade : Picking up the grenade power up instantly wipes out ever enemy presently on the screen, including Armor Tanks regardless of how many times you've hit them. You do not, however, get credit for destroying them during the end-stage bonus points.
        helmet  : The helmet power up grants you a temporary force field that makes you invulnerable to enemy shots, just like the one you begin every stage with.
        shovel  : The shovel power up turns the walls around your fortress from brick to stone. This makes it impossible for the enemy to penetrate the wall and destroy your fortress, ending the game prematurely. The effect, however, is only temporary, and will wear off eventually.
        star        : The star power up grants your tank with new offensive power each time you pick one up, up to three times. The first star allows you to fire your bullets as fast as the power tanks can. The second star allows you to fire up to two bullets on the screen at one time. And the third star allows your bullets to destroy the otherwise unbreakable steel walls. You carry this power with you to each new stage until you lose a life.
        tank        : The tank power up grants you one extra life. The only other way to get an extra life is to score 20000 points.
        timer       : The timer power up temporarily freezes time, allowing you to harmlessly approach every tank and destroy them until the time freeze wears off.
    """

    # bonus types
    (BONUS_GRENADE, BONUS_HELMET, BONUS_SHOVEL, BONUS_STAR, BONUS_TANK, BONUS_TIMER) = range(6)

    def __init__(self, level):

        global sprites

        # to know where to place
        self.level = level

        # bonus lives only for a limited period of time
        self.active = True

        # blinking state
        self.visible = True

        self.rect = pygame.Rect(random.randint(0, 416-32), random.randint(0, 416-32), 32, 32)

        self.bonus = random.choice([
            self.BONUS_GRENADE,
            self.BONUS_HELMET,
            self.BONUS_SHOVEL,
            self.BONUS_STAR,
            self.BONUS_TANK,
            self.BONUS_TIMER
        ])

        self.image = sprites.subsurface(16*2*self.bonus, 32*2, 16*2, 15*2)

    def draw(self):
        """ draw bonus """
        global screen
        if self.visible:
            screen.blit(self.image, self.rect.topleft)

    def toggleVisibility(self):
        """ Toggle bonus visibility """
        self.visible = not self.visible


class Bullet():
    # direction constants
    (DIR_UP, DIR_RIGHT, DIR_DOWN, DIR_LEFT) = range(4)

    # bullet's stated
    (STATE_REMOVED, STATE_ACTIVE, STATE_EXPLODING) = range(3)

    (OWNER_PLAYER, OWNER_ENEMY) = range(2)

    def __init__(self, level, position, direction, damage = 100, speed = 5):

        global sprites

        self.level = level
        self.direction = direction
        self.damage = damage
        self.owner = None
        self.owner_class = None

        # 1-regular everyday normal bullet
        # 2-can destroy steel
        self.power = 1

        self.image = sprites.subsurface(75*2, 74*2, 3*2, 4*2)

        # position is player's top left corner, so we'll need to
        # recalculate a bit. also rotate image itself.
        if direction == self.DIR_UP:
            self.rect = pygame.Rect(position[0] + 11, position[1] - 8, 6, 8)
        elif direction == self.DIR_RIGHT:
            self.image = pygame.transform.rotate(self.image, 270)
            self.rect = pygame.Rect(position[0] + 26, position[1] + 11, 8, 6)
        elif direction == self.DIR_DOWN:
            self.image = pygame.transform.rotate(self.image, 180)
            self.rect = pygame.Rect(position[0] + 11, position[1] + 26, 6, 8)
        elif direction == self.DIR_LEFT:
            self.image = pygame.transform.rotate(self.image, 90)
            self.rect = pygame.Rect(position[0] - 8 , position[1] + 11, 8, 6)

        self.explosion_images = [
            sprites.subsurface(0, 80*2, 32*2, 32*2),
            sprites.subsurface(32*2, 80*2, 32*2, 32*2),
        ]

        self.speed = speed

        self.state = self.STATE_ACTIVE

    def draw(self):
        """ draw bullet """
        global screen
        if self.state == self.STATE_ACTIVE:
            screen.blit(self.image, self.rect.topleft)
        elif self.state == self.STATE_EXPLODING:
            self.explosion.draw()

    def update(self):
        global castle, players, enemies, bullets

        if self.state == self.STATE_EXPLODING:
            if not self.explosion.active:
                self.destroy()
                del self.explosion

        if self.state != self.STATE_ACTIVE:
            return

        """ move bullet """
        if self.direction == self.DIR_UP:
            self.rect.topleft = [self.rect.left, self.rect.top - self.speed]
            if self.rect.top < 0:
                self.explode()
                return
        elif self.direction == self.DIR_RIGHT:
            self.rect.topleft = [self.rect.left + self.speed, self.rect.top]
            if self.rect.left > (416 - self.rect.width):
                self.explode()
                return
        elif self.direction == self.DIR_DOWN:
            self.rect.topleft = [self.rect.left, self.rect.top + self.speed]
            if self.rect.top > (416 - self.rect.height):
                self.explode()
                return
        elif self.direction == self.DIR_LEFT:
            self.rect.topleft = [self.rect.left - self.speed, self.rect.top]
            if self.rect.left < 0:
                self.explode()
                return

        has_collided = False

        # check for collisions with walls. one bullet can destroy several (1 or 2)
        # tiles but explosion remains 1
        rects = self.level.obstacle_rects
        collisions = self.rect.collidelistall(rects)
        if collisions != []:
            for i in collisions:
                if self.level.hitTile(rects[i].topleft, self.power, self.owner == self.OWNER_PLAYER):
                    has_collided = True
        if has_collided:
            self.explode()
            return

        # check for collisions with other bullets
        for bullet in bullets:
            if self.state == self.STATE_ACTIVE and bullet.owner != self.owner and bullet != self and self.rect.colliderect(bullet.rect):
                self.destroy()
                self.explode()
                return

        # check for collisions with players
        for player in players:
            if player.state == player.STATE_ALIVE and self.rect.colliderect(player.rect):
                if player.bulletImpact(self.owner == self.OWNER_PLAYER, self.damage, self.owner_class):
                    self.destroy()
                    return

        # check for collisions with enemies
        for enemy in enemies:
            if enemy.state == enemy.STATE_ALIVE and self.rect.colliderect(enemy.rect):
                if enemy.bulletImpact(self.owner == self.OWNER_ENEMY, self.damage, self.owner_class):
                    self.destroy()
                    return

        # check for collision with castle
        if castle.active and self.rect.colliderect(castle.rect):
            castle.destroy()
            self.destroy()
            return

    def explode(self):
        """ start bullets's explosion """
        global screen
        if self.state != self.STATE_REMOVED:
            self.state = self.STATE_EXPLODING
            self.explosion = Explosion([self.rect.left-13, self.rect.top-13], None, self.explosion_images)

    def destroy(self):
        self.state = self.STATE_REMOVED


class Label():
    def __init__(self, position, text = "", duration = None):

        self.position = position

        self.active = True

        self.text = text

        self.font = pygame.font.SysFont("Arial", 13)

        if duration != None:
            gtimer.add(duration, lambda :self.destroy(), 1)

    def draw(self):
        """ draw label """
        global screen
        screen.blit(self.font.render(self.text, False, (200,200,200)), [self.position[0]+4, self.position[1]+8])

    def destroy(self):
        self.active = False


class Explosion():
    def __init__(self, position, interval = None, images = None):

        global sprites

        self.position = [position[0]-16, position[1]-16]
        self.active = True

        if interval == None:
            interval = 100

        if images == None:
            images = [
                sprites.subsurface(0, 80*2, 32*2, 32*2),
                sprites.subsurface(32*2, 80*2, 32*2, 32*2),
                sprites.subsurface(64*2, 80*2, 32*2, 32*2)
            ]

        images.reverse()

        self.images = [] + images

        self.image = self.images.pop()

        gtimer.add(interval, lambda :self.update(), len(self.images) + 1)

    def draw(self):
        global screen
        """ draw current explosion frame """
        screen.blit(self.image, self.position)

    def update(self):
        """ Advace to the next image """
        if len(self.images) > 0:
            self.image = self.images.pop()
        else:
            self.active = False

class Level():

    # tile constants
    (TILE_EMPTY, TILE_BRICK, TILE_STEEL, TILE_WATER, TILE_GRASS, TILE_FROZE) = range(6)

    # tile width/height in px
    TILE_SIZE = 16

    def __init__(self, level_nr = None):
        """ There are total 35 different levels. If level_nr is larger than 35, loop over
        to next according level so, for example, if level_nr ir 37, then load level 2 """

        global sprites

        # max number of enemies simultaneously  being on map
        self.max_active_enemies = 4

        tile_images = [
            pygame.Surface((8*2, 8*2)),
            sprites.subsurface(48*2, 64*2, 8*2, 8*2),
            sprites.subsurface(48*2, 72*2, 8*2, 8*2),
            sprites.subsurface(56*2, 72*2, 8*2, 8*2),
            sprites.subsurface(64*2, 64*2, 8*2, 8*2),
            sprites.subsurface(64*2, 64*2, 8*2, 8*2),
            sprites.subsurface(72*2, 64*2, 8*2, 8*2),
            sprites.subsurface(64*2, 72*2, 8*2, 8*2)
        ]
        self.tile_empty = tile_images[0]
        self.tile_brick = tile_images[1]
        self.tile_steel = tile_images[2]
        self.tile_grass = tile_images[3]
        self.tile_water = tile_images[4]
        self.tile_water1= tile_images[4]
        self.tile_water2= tile_images[5]
        self.tile_froze = tile_images[6]

        self.obstacle_rects = []

        level_nr = 1 if level_nr == None else level_nr%35
        if level_nr == 0:
            level_nr = 35

        self.loadLevel(level_nr)

        # tiles' rects on map, tanks cannot move over
        self.obstacle_rects = []

        # update these tiles
        self.updateObstacleRects()

        gtimer.add(400, lambda :self.toggleWaves())

    def hitTile(self, pos, power = 1, sound = False):
        """
            Hit the tile
            @param pos Tile's x, y in px
            @return True if bullet was stopped, False otherwise
        """

        for tile in self.mapr:
            if tile.topleft == pos:
                if tile.type == self.TILE_BRICK:
                    self.mapr.remove(tile)
                    self.updateObstacleRects()
                    return True
                elif tile.type == self.TILE_STEEL:
                    if power == 2:
                        self.mapr.remove(tile)
                        self.updateObstacleRects()
                    return True
                else:
                    return False

    def toggleWaves(self):
        """ Toggle water image """
        if self.tile_water == self.tile_water1:
            self.tile_water = self.tile_water2
        else:
            self.tile_water = self.tile_water1


    def loadLevel(self, level_nr = 1):
        """ Load specified level
        @return boolean Whether level was loaded
        """
        filename = "tanksbattle/levels/"+str(level_nr)
        if (not os.path.isfile(filename)):
            return False
        level = []
        f = open(filename, "r")
        data = f.read().split("\n")
        self.mapr = []
        x, y = 0, 0
        for row in data:
            for ch in row:
                if ch == "#":
                    self.mapr.append(myRect(x, y, self.TILE_SIZE, self.TILE_SIZE, self.TILE_BRICK))
                elif ch == "@":
                    self.mapr.append(myRect(x, y, self.TILE_SIZE, self.TILE_SIZE, self.TILE_STEEL))
                elif ch == "~":
                    self.mapr.append(myRect(x, y, self.TILE_SIZE, self.TILE_SIZE, self.TILE_WATER))
                elif ch == "%":
                    self.mapr.append(myRect(x, y, self.TILE_SIZE, self.TILE_SIZE, self.TILE_GRASS))
                elif ch == "-":
                    self.mapr.append(myRect(x, y, self.TILE_SIZE, self.TILE_SIZE, self.TILE_FROZE))
                x += self.TILE_SIZE
            x = 0
            y += self.TILE_SIZE
        return True


    def draw(self, tiles = None):
        """ Draw specified map on top of existing surface """

        global screen

        if tiles == None:
            tiles = [TILE_BRICK, TILE_STEEL, TILE_WATER, TILE_GRASS, TILE_FROZE]

        for tile in self.mapr:
            if tile.type in tiles:
                if tile.type == self.TILE_BRICK:
                    screen.blit(self.tile_brick, tile.topleft)
                elif tile.type == self.TILE_STEEL:
                    screen.blit(self.tile_steel, tile.topleft)
                elif tile.type == self.TILE_WATER:
                    screen.blit(self.tile_water, tile.topleft)
                elif tile.type == self.TILE_FROZE:
                    screen.blit(self.tile_froze, tile.topleft)
                elif tile.type == self.TILE_GRASS:
                    screen.blit(self.tile_grass, tile.topleft)

    def updateObstacleRects(self):
        """ Set self.obstacle_rects to all tiles' rects that players can destroy
        with bullets """

        global castle

        self.obstacle_rects = [castle.rect]

        for tile in self.mapr:
            if tile.type in (self.TILE_BRICK, self.TILE_STEEL, self.TILE_WATER):
                self.obstacle_rects.append(tile)

    def buildFortress(self, tile):
        """ Build walls around castle made from tile """

        positions = [
            (11*self.TILE_SIZE, 23*self.TILE_SIZE),
            (11*self.TILE_SIZE, 24*self.TILE_SIZE),
            (11*self.TILE_SIZE, 25*self.TILE_SIZE),
            (14*self.TILE_SIZE, 23*self.TILE_SIZE),
            (14*self.TILE_SIZE, 24*self.TILE_SIZE),
            (14*self.TILE_SIZE, 25*self.TILE_SIZE),
            (12*self.TILE_SIZE, 23*self.TILE_SIZE),
            (13*self.TILE_SIZE, 23*self.TILE_SIZE)
        ]

        obsolete = []

        for i, rect in enumerate(self.mapr):
            if rect.topleft in positions:
                obsolete.append(rect)
        for rect in obsolete:
            self.mapr.remove(rect)

        for pos in positions:
            self.mapr.append(myRect(pos[0], pos[1], self.TILE_SIZE, self.TILE_SIZE, tile))

        self.updateObstacleRects()

class Tank():

    # possible directions
    (DIR_UP, DIR_RIGHT, DIR_DOWN, DIR_LEFT) = range(4)

    # states
    (STATE_SPAWNING, STATE_DEAD, STATE_ALIVE, STATE_EXPLODING) = range(4)

    # sides
    (SIDE_PLAYER, SIDE_ENEMY) = range(2)

    def __init__(self, level, side, position = None, direction = None, filename = None):

        global sprites

        # health. 0 health means dead
        self.health = 100

        # tank can't move but can rotate and shoot
        self.paralised = False

        # tank can't do anything
        self.paused = False

        # tank is protected from bullets
        self.shielded = False

        # px per move
        self.speed = 2

        # how many bullets can tank fire simultaneously
        self.max_active_bullets = 1

        # friend or foe
        self.side = side

        # flashing state. 0-off, 1-on
        self.flash = 0

        # 0 - no superpowers
        # 1 - faster bullets
        # 2 - can fire 2 bullets
        # 3 - can destroy steel
        self.superpowers = 0

        # each tank can pick up 1 bonus
        self.bonus = None

        # navigation keys: fire, up, right, down, left
        self.controls = [pygame.K_SPACE, pygame.K_UP, pygame.K_RIGHT, pygame.K_DOWN, pygame.K_LEFT]

        # currently pressed buttons (navigation only)
        self.pressed = [False] * 4

        self.shield_images = [
            sprites.subsurface(0, 48*2, 16*2, 16*2),
            sprites.subsurface(16*2, 48*2, 16*2, 16*2)
        ]
        self.shield_image = self.shield_images[0]
        self.shield_index = 0

        self.spawn_images = [
            sprites.subsurface(32*2, 48*2, 16*2, 16*2),
            sprites.subsurface(48*2, 48*2, 16*2, 16*2)
        ]
        self.spawn_image = self.spawn_images[0]
        self.spawn_index = 0

        self.level = level

        if position != None:
            self.rect = pygame.Rect(position, (26, 26))
        else:
            self.rect = pygame.Rect(0, 0, 26, 26)

        if direction == None:
            self.direction = random.choice([self.DIR_RIGHT, self.DIR_DOWN, self.DIR_LEFT])
        else:
            self.direction = direction

        self.state = self.STATE_SPAWNING

        # spawning animation
        self.timer_uuid_spawn = gtimer.add(100, lambda :self.toggleSpawnImage())

        # duration of spawning
        self.timer_uuid_spawn_end = gtimer.add(1000, lambda :self.endSpawning())

    def endSpawning(self):
        """ End spawning
        Player becomes operational
        """
        self.state = self.STATE_ALIVE
        gtimer.destroy(self.timer_uuid_spawn_end)


    def toggleSpawnImage(self):
        """ advance to the next spawn image """
        if self.state != self.STATE_SPAWNING:
            gtimer.destroy(self.timer_uuid_spawn)
            return
        self.spawn_index += 1
        if self.spawn_index >= len(self.spawn_images):
            self.spawn_index = 0
        self.spawn_image = self.spawn_images[self.spawn_index]

    def toggleShieldImage(self):
        """ advance to the next shield image """
        if self.state != self.STATE_ALIVE:
            gtimer.destroy(self.timer_uuid_shield)
            return
        if self.shielded:
            self.shield_index += 1
            if self.shield_index >= len(self.shield_images):
                self.shield_index = 0
            self.shield_image = self.shield_images[self.shield_index]


    def draw(self):
        """ draw tank """
        global screen
        if self.state == self.STATE_ALIVE:
            screen.blit(self.image, self.rect.topleft)
            if self.shielded:
                screen.blit(self.shield_image, [self.rect.left-3, self.rect.top-3])
        elif self.state == self.STATE_EXPLODING:
            self.explosion.draw()
        elif self.state == self.STATE_SPAWNING:
            screen.blit(self.spawn_image, self.rect.topleft)

    def explode(self):
        """ start tanks's explosion """
        if self.state != self.STATE_DEAD:
            self.state = self.STATE_EXPLODING
            self.explosion = Explosion(self.rect.topleft)
            if self.bonus:
                self.spawnBonus()

    def fire(self, forced = False):
        """ Shoot a bullet
        @param boolean forced. If false, check whether tank has exceeded his bullet quota. Default: True
        @return boolean True if bullet was fired, false otherwise
        """

        global bullets, labels

        if self.state != self.STATE_ALIVE:
            gtimer.destroy(self.timer_uuid_fire)
            return False

        if self.paused:
            return False

        if not forced:
            active_bullets = 0
            for bullet in bullets:
                if bullet.owner_class == self and bullet.state == bullet.STATE_ACTIVE:
                    active_bullets += 1
            if active_bullets >= self.max_active_bullets:
                return False

        bullet = Bullet(self.level, self.rect.topleft, self.direction)

        # if superpower level is at least 1
        if self.superpowers > 0:
            bullet.speed = 8

        # if superpower level is at least 3
        if self.superpowers > 2:
            bullet.power = 2

        if self.side == self.SIDE_PLAYER:
            bullet.owner = self.SIDE_PLAYER
        else:
            bullet.owner = self.SIDE_ENEMY
            self.bullet_queued = False

        bullet.owner_class = self
        bullets.append(bullet)
        return True

    def rotate(self, direction, fix_position = True):
        """ Rotate tank
        rotate, update image and correct position
        """
        self.direction = direction

        if direction == self.DIR_UP:
            self.image = self.image_up
        elif direction == self.DIR_RIGHT:
            self.image = self.image_right
        elif direction == self.DIR_DOWN:
            self.image = self.image_down
        elif direction == self.DIR_LEFT:
            self.image = self.image_left

        if fix_position:
            new_x = self.nearest(self.rect.left, 8) + 3
            new_y = self.nearest(self.rect.top, 8) + 3

            if (abs(self.rect.left - new_x) < 5):
                self.rect.left = new_x

            if (abs(self.rect.top - new_y) < 5):
                self.rect.top = new_y

    def turnAround(self):
        """ Turn tank into opposite direction """
        if self.direction in (self.DIR_UP, self.DIR_RIGHT):
            self.rotate(self.direction + 2, False)
        else:
            self.rotate(self.direction - 2, False)

    def update(self, time_passed):
        """ Update timer and explosion (if any) """
        if self.state == self.STATE_EXPLODING:
            if not self.explosion.active:
                self.state = self.STATE_DEAD
                del self.explosion

    def nearest(self, num, base):
        """ Round number to nearest divisible """
        return int(round(num / (base * 1.0)) * base)


    def bulletImpact(self, friendly_fire = False, damage = 100, tank = None):
        """ Bullet impact
        Return True if bullet should be destroyed on impact. Only enemy friendly-fire
        doesn't trigger bullet explosion
        """

        if self.shielded:
            return True

        if not friendly_fire:
            self.health -= damage
            if self.health < 1:
                if self.side == self.SIDE_ENEMY:
                    tank.trophies["enemy"+str(self.type)] += 1
                    points = (self.type+1) * 100
                    tank.score += points

                    labels.append(Label(self.rect.topleft, str(points), 500))
                self.explode()
            return True

        if self.side == self.SIDE_ENEMY:
            return False
        elif self.side == self.SIDE_PLAYER:
            if not self.paralised:
                self.setParalised(True)
                self.timer_uuid_paralise = gtimer.add(10000, lambda :self.setParalised(False), 1)
            return True

    def setParalised(self, paralised = True):
        """ set tank paralise state
        @param boolean paralised
        @return None
        """
        if self.state != self.STATE_ALIVE:
            gtimer.destroy(self.timer_uuid_paralise)
            return
        self.paralised = paralised

class Enemy(Tank):

    (TYPE_BASIC, TYPE_FAST, TYPE_POWER, TYPE_ARMOR) = range(4)

    def __init__(self, level, type, position = None, direction = None, filename = None):

        Tank.__init__(self, level, type, position = None, direction = None, filename = None)

        global enemies, sprites

        # if true, do not fire
        self.bullet_queued = False

        # chose type on random
        if len(level.enemies_left) > 0:
            self.type = level.enemies_left.pop()
        else:
            self.state = self.STATE_DEAD
            return

        if self.type == self.TYPE_BASIC:
            self.speed = 1
        elif self.type == self.TYPE_FAST:
            self.speed = 3
        elif self.type == self.TYPE_POWER:
            self.superpowers = 1
        elif self.type == self.TYPE_ARMOR:
            self.health = 400

        # 1 in 5 chance this will be bonus carrier, but only if no other tank is
        if random.randint(1, 5) == 1:
            self.bonus = True
            for enemy in enemies:
                if enemy.bonus:
                    self.bonus = False
                    break

        images = [
            sprites.subsurface(32*2, 0, 13*2, 15*2),
            sprites.subsurface(48*2, 0, 13*2, 15*2),
            sprites.subsurface(64*2, 0, 13*2, 15*2),
            sprites.subsurface(80*2, 0, 13*2, 15*2),
            sprites.subsurface(32*2, 16*2, 13*2, 15*2),
            sprites.subsurface(48*2, 16*2, 13*2, 15*2),
            sprites.subsurface(64*2, 16*2, 13*2, 15*2),
            sprites.subsurface(80*2, 16*2, 13*2, 15*2)
        ]

        self.image = images[self.type+0]

        self.image_up = self.image;
        self.image_left = pygame.transform.rotate(self.image, 90)
        self.image_down = pygame.transform.rotate(self.image, 180)
        self.image_right = pygame.transform.rotate(self.image, 270)

        if self.bonus:
            self.image1_up = self.image_up;
            self.image1_left = self.image_left
            self.image1_down = self.image_down
            self.image1_right = self.image_right

            self.image2 = images[self.type+4]
            self.image2_up = self.image2;
            self.image2_left = pygame.transform.rotate(self.image2, 90)
            self.image2_down = pygame.transform.rotate(self.image2, 180)
            self.image2_right = pygame.transform.rotate(self.image2, 270)

        self.rotate(self.direction, False)

        if position == None:
            self.rect.topleft = self.getFreeSpawningPosition()
            if not self.rect.topleft:
                self.state = self.STATE_DEAD
                return

        # list of map coords where tank should go next
        self.path = self.generatePath(self.direction)

        # 1000 is duration between shots
        self.timer_uuid_fire = gtimer.add(1000, lambda :self.fire())

        # turn on flashing
        if self.bonus:
            self.timer_uuid_flash = gtimer.add(200, lambda :self.toggleFlash())

    def toggleFlash(self):
        """ Toggle flash state """
        if self.state not in (self.STATE_ALIVE, self.STATE_SPAWNING):
            gtimer.destroy(self.timer_uuid_flash)
            return
        self.flash = not self.flash
        if self.flash:
            self.image_up = self.image2_up
            self.image_right = self.image2_right
            self.image_down = self.image2_down
            self.image_left = self.image2_left
        else:
            self.image_up = self.image1_up
            self.image_right = self.image1_right
            self.image_down = self.image1_down
            self.image_left = self.image1_left
        self.rotate(self.direction, False)

    def spawnBonus(self):
        """ Create new bonus if needed """

        global bonuses

        if len(bonuses) > 0:
            return
        bonus = Bonus(self.level)
        bonuses.append(bonus)
        gtimer.add(500, lambda :bonus.toggleVisibility())
        gtimer.add(10000, lambda :bonuses.remove(bonus), 1)


    def getFreeSpawningPosition(self):

        global players, enemies

        available_positions = [
            [(self.level.TILE_SIZE * 2 - self.rect.width) / 2, (self.level.TILE_SIZE * 2 - self.rect.height) / 2],
            [12 * self.level.TILE_SIZE + (self.level.TILE_SIZE * 2 - self.rect.width) / 2, (self.level.TILE_SIZE * 2 - self.rect.height) / 2],
            [24 * self.level.TILE_SIZE + (self.level.TILE_SIZE * 2 - self.rect.width) / 2,  (self.level.TILE_SIZE * 2 - self.rect.height) / 2]
        ]

        random.shuffle(available_positions)

        for pos in available_positions:

            enemy_rect = pygame.Rect(pos, [26, 26])

            # collisions with other enemies
            collision = False
            for enemy in enemies:
                if enemy_rect.colliderect(enemy.rect):
                    collision = True
                    continue

            if collision:
                continue

            # collisions with players
            collision = False
            for player in players:
                if enemy_rect.colliderect(player.rect):
                    collision = True
                    continue

            if collision:
                continue

            return pos
        return False

    def move(self):
        """ move enemy if possible """

        global players, enemies, bonuses

        if self.state != self.STATE_ALIVE or self.paused or self.paralised:
            return

        if self.path == []:
            self.path = self.generatePath(None, True)

        new_position = self.path.pop(0)

        # move enemy
        if self.direction == self.DIR_UP:
            if new_position[1] < 0:
                self.path = self.generatePath(self.direction, True)
                return
        elif self.direction == self.DIR_RIGHT:
            if new_position[0] > (416 - 26):
                self.path = self.generatePath(self.direction, True)
                return
        elif self.direction == self.DIR_DOWN:
            if new_position[1] > (416 - 26):
                self.path = self.generatePath(self.direction, True)
                return
        elif self.direction == self.DIR_LEFT:
            if new_position[0] < 0:
                self.path = self.generatePath(self.direction, True)
                return

        new_rect = pygame.Rect(new_position, [26, 26])

        # collisions with tiles
        if new_rect.collidelist(self.level.obstacle_rects) != -1:
            self.path = self.generatePath(self.direction, True)
            return

        # collisions with other enemies
        for enemy in enemies:
            if enemy != self and new_rect.colliderect(enemy.rect):
                self.turnAround()
                self.path = self.generatePath(self.direction)
                return

        # collisions with players
        for player in players:
            if new_rect.colliderect(player.rect):
                self.turnAround()
                self.path = self.generatePath(self.direction)
                return

        # collisions with bonuses
        for bonus in bonuses:
            if new_rect.colliderect(bonus.rect):
                bonuses.remove(bonus)

        # if no collision, move enemy
        self.rect.topleft = new_rect.topleft


    def update(self, time_passed):
        Tank.update(self, time_passed)
        if self.state == self.STATE_ALIVE and not self.paused:
            self.move()

    def generatePath(self, direction = None, fix_direction = False):
        """ If direction is specified, try continue that way, otherwise choose at random
        """

        all_directions = [self.DIR_UP, self.DIR_RIGHT, self.DIR_DOWN, self.DIR_LEFT]

        if direction == None:
            if self.direction in [self.DIR_UP, self.DIR_RIGHT]:
                opposite_direction = self.direction + 2
            else:
                opposite_direction = self.direction - 2
            directions = all_directions
            random.shuffle(directions)
            directions.remove(opposite_direction)
            directions.append(opposite_direction)
        else:
            if direction in [self.DIR_UP, self.DIR_RIGHT]:
                opposite_direction = direction + 2
            else:
                opposite_direction = direction - 2

            if direction in [self.DIR_UP, self.DIR_RIGHT]:
                opposite_direction = direction + 2
            else:
                opposite_direction = direction - 2
            directions = all_directions
            random.shuffle(directions)
            directions.remove(opposite_direction)
            directions.remove(direction)
            directions.insert(0, direction)
            directions.append(opposite_direction)

        # at first, work with general units (steps) not px
        x = int(round(self.rect.left / 16))
        y = int(round(self.rect.top / 16))

        new_direction = None

        for direction in directions:
            if direction == self.DIR_UP and y > 1:
                new_pos_rect = self.rect.move(0, -8)
                if new_pos_rect.collidelist(self.level.obstacle_rects) == -1:
                    new_direction = direction
                    break
            elif direction == self.DIR_RIGHT and x < 24:
                new_pos_rect = self.rect.move(8, 0)
                if new_pos_rect.collidelist(self.level.obstacle_rects) == -1:
                    new_direction = direction
                    break
            elif direction == self.DIR_DOWN and y < 24:
                new_pos_rect = self.rect.move(0, 8)
                if new_pos_rect.collidelist(self.level.obstacle_rects) == -1:
                    new_direction = direction
                    break
            elif direction == self.DIR_LEFT and x > 1:
                new_pos_rect = self.rect.move(-8, 0)
                if new_pos_rect.collidelist(self.level.obstacle_rects) == -1:
                    new_direction = direction
                    break

        # if we can go anywhere else, turn around
        if new_direction == None:
            new_direction = opposite_direction
            print("nav izejas. griezhamies")

        # fix tanks position
        if fix_direction and new_direction == self.direction:
            fix_direction = False

        self.rotate(new_direction, fix_direction)

        positions = []

        x = self.rect.left
        y = self.rect.top

        if new_direction in (self.DIR_RIGHT, self.DIR_LEFT):
            axis_fix = self.nearest(y, 16) - y
        else:
            axis_fix = self.nearest(x, 16) - x
        axis_fix = 0

        pixels = self.nearest(random.randint(1, 12) * 32, 32) + axis_fix + 3

        if new_direction == self.DIR_UP:
            for px in range(0, pixels, self.speed):
                positions.append([x, y-px])
        elif new_direction == self.DIR_RIGHT:
            for px in range(0, pixels, self.speed):
                positions.append([x+px, y])
        elif new_direction == self.DIR_DOWN:
            for px in range(0, pixels, self.speed):
                positions.append([x, y+px])
        elif new_direction == self.DIR_LEFT:
            for px in range(0, pixels, self.speed):
                positions.append([x-px, y])

        return positions



class Player(Tank):

    def __init__(self, level, type, position = None, direction = None, filename = None):

        Tank.__init__(self, level, type, position = None, direction = None, filename = None)

        global sprites

        if filename == None:
            filename = (0, 0, 16*2, 16*2)

        self.start_position = position
        self.start_direction = direction

        self.lives = 3

        # total score
        self.score = 0

        # store how many bonuses in this stage this player has collected
        self.trophies = {
            "bonus" : 0,
            "enemy0" : 0,
            "enemy1" : 0,
            "enemy2" : 0,
            "enemy3" : 0
        }

        self.image = sprites.subsurface(filename)
        self.image_up = self.image;
        self.image_left = pygame.transform.rotate(self.image, 90)
        self.image_down = pygame.transform.rotate(self.image, 180)
        self.image_right = pygame.transform.rotate(self.image, 270)

        if direction == None:
            self.rotate(self.DIR_UP, False)
        else:
            self.rotate(direction, False)

    def move(self, direction):
        """ move player if possible """

        global players, enemies, bonuses

        if self.state == self.STATE_EXPLODING:
            if not self.explosion.active:
                self.state = self.STATE_DEAD
                del self.explosion

        if self.state != self.STATE_ALIVE:
            return

        # rotate player
        if self.direction != direction:
            self.rotate(direction)

        if self.paralised:
            return

        # move player
        if direction == self.DIR_UP:
            new_position = [self.rect.left, self.rect.top - self.speed]
            if new_position[1] < 0:
                return
        elif direction == self.DIR_RIGHT:
            new_position = [self.rect.left + self.speed, self.rect.top]
            if new_position[0] > (416 - 26):
                return
        elif direction == self.DIR_DOWN:
            new_position = [self.rect.left, self.rect.top + self.speed]
            if new_position[1] > (416 - 26):
                return
        elif direction == self.DIR_LEFT:
            new_position = [self.rect.left - self.speed, self.rect.top]
            if new_position[0] < 0:
                return

        player_rect = pygame.Rect(new_position, [26, 26])

        # collisions with tiles
        if player_rect.collidelist(self.level.obstacle_rects) != -1:
            return

        # collisions with other players
        for player in players:
            if player != self and player.state == player.STATE_ALIVE and player_rect.colliderect(player.rect) == True:
                return

        # collisions with enemies
        for enemy in enemies:
            if player_rect.colliderect(enemy.rect) == True:
                return

        # collisions with bonuses
        for bonus in bonuses:
            if player_rect.colliderect(bonus.rect) == True:
                self.bonus = bonus

        #if no collision, move player
        self.rect.topleft = (new_position[0], new_position[1])

    def reset(self):
        """ reset player """
        self.rotate(self.start_direction, False)
        self.rect.topleft = self.start_position
        self.superpowers = 0
        self.max_active_bullets = 1
        self.health = 100
        self.paralised = False
        self.paused = False
        self.pressed = [False] * 4
        self.state = self.STATE_ALIVE


class GameState:
    # direction constants
    (DIR_UP, DIR_RIGHT, DIR_DOWN, DIR_LEFT) = range(4)

    TILE_SIZE = 16
    

    def __init__(self):

        global screen, sprites, castle, gtimer

        # center window
        os.environ['SDL_VIDEO_WINDOW_POS'] = 'center'

        pygame.init()


        pygame.display.set_caption("Battle City")

        size = width, height = 480, 416
        screen = pygame.display.set_mode(size)

        self.clock = pygame.time.Clock()

        # load sprites (funky version)
        #sprites = pygame.transform.scale2x(pygame.image.load("images/sprites.gif"))
        # load sprites (pixely version)
        sprites = pygame.transform.scale(pygame.image.load("tanksbattle/images/sprites.gif"), [192, 224])
        #screen.set_colorkey((0,138,104))
        castle = Castle()
        gtimer = Timer()

        pygame.display.set_icon(sprites.subsurface(0, 0, 13*2, 13*2))

        self.enemy_life_image = sprites.subsurface(81*2, 57*2, 7*2, 7*2)
        self.player_life_image = sprites.subsurface(89*2, 56*2, 7*2, 8*2)
        self.flag_image = sprites.subsurface(64*2, 49*2, 16*2, 15*2)

        # this is used in intro screen
        self.player_image = pygame.transform.rotate(sprites.subsurface(0, 0, 13*2, 13*2), 270)

        # if true, no new enemies will be spawn during this time
        self.timefreeze = False

        # load custom font
        self.font = pygame.font.Font("tanksbattle/fonts/prstart.ttf", 16)

        # number of players. here is defined preselected menu value
        self.nr_of_players = 1

        del players[:]
        del bullets[:]
        del enemies[:]
        del bonuses[:]

        self.nextLevel()

    def nextLevel(self):
        """ Start next level """

        global castle, players, bullets, bonuses

        del bullets[:]
        del enemies[:]
        del bonuses[:]
        castle.rebuild()
        del gtimer.timers[:]
        del labels[:]

        # load 1 level
        self.stage = GAME_LEVEL
        self.level = Level(self.stage)
        self.timefreeze = False

        # set number of enemies by types (basic, fast, power, armor) according to level
        levels_enemies = (
            (18,2,0,0), (14,4,0,2), (14,4,0,2), (2,5,10,3), (8,5,5,2),
            (9,2,7,2), (7,4,6,3), (7,4,7,2), (6,4,7,3), (12,2,4,2),
            (5,5,4,6), (0,6,8,6), (0,8,8,4), (0,4,10,6), (0,2,10,8),
            (16,2,0,2), (8,2,8,2), (2,8,6,4), (4,4,4,8), (2,8,2,8),
            (6,2,8,4), (6,8,2,4), (0,10,4,6), (10,4,4,2), (0,8,2,10),
            (4,6,4,6), (2,8,2,8), (15,2,2,1), (0,4,10,6), (4,8,4,4),
            (3,8,3,6), (6,4,2,8), (4,4,4,8), (0,10,4,6), (0,6,4,10)
        )
        

        if self.stage <= 35:
            enemies_l = levels_enemies[self.stage - 1]
        else:
            enemies_l = levels_enemies[34]

        self.level.enemies_left = [0]*enemies_l[0] + [1]*enemies_l[1] + [2]*enemies_l[2] + [3]*enemies_l[3]
        random.shuffle(self.level.enemies_left)

        self.reloadPlayers()

        gtimer.add(3000, lambda :self.spawnEnemy())

        # if True, start "game over" animation
        self.game_over = False

        # if False, game will end w/o "game over" bussiness
        self.running = True

        # if False, players won't be able to do anything
        self.active = True

        self.draw()

    def triggerBonus(self, bonus, player):
        """ Execute bonus powers """

        global enemies, labels

        player.trophies["bonus"] += 1
        player.score += 500

        if bonus.bonus == bonus.BONUS_GRENADE:
            for enemy in enemies:
                enemy.explode()
        elif bonus.bonus == bonus.BONUS_HELMET:
            self.shieldPlayer(player, True, 10000)
        elif bonus.bonus == bonus.BONUS_SHOVEL:
            self.level.buildFortress(self.level.TILE_STEEL)
            gtimer.add(10000, lambda :self.level.buildFortress(self.level.TILE_BRICK), 1)
        elif bonus.bonus == bonus.BONUS_STAR:
            player.superpowers += 1
            if player.superpowers == 2:
                player.max_active_bullets = 2
        elif bonus.bonus == bonus.BONUS_TANK:
            player.lives += 1
        elif bonus.bonus == bonus.BONUS_TIMER:
            self.toggleEnemyFreeze(True)
            gtimer.add(10000, lambda :self.toggleEnemyFreeze(False), 1)
        bonuses.remove(bonus)

        labels.append(Label(bonus.rect.topleft, "500", 500))

    def shieldPlayer(self, player, shield = True, duration = None):
        """ Add/remove shield
        player: player (not enemy)
        shield: true/false
        duration: in ms. if none, do not remove shield automatically
        """
        player.shielded = shield
        if shield:
            player.timer_uuid_shield = gtimer.add(100, lambda :player.toggleShieldImage())
        else:
            gtimer.destroy(player.timer_uuid_shield)

        if shield and duration != None:
            gtimer.add(duration, lambda :self.shieldPlayer(player, False), 1)


    def spawnEnemy(self):
        """ Spawn new enemy if needed
        Only add enemy if:
            - there are at least one in queue
            - map capacity hasn't exceeded its quota
            - now isn't timefreeze
        """

        global enemies
        if len(enemies) >= self.level.max_active_enemies:
            return
        if len(self.level.enemies_left) < 1 or self.timefreeze:
            return
        enemy = Enemy(self.level, 1)

        enemies.append(enemy)


    def respawnPlayer(self, player, clear_scores = False):
        """ Respawn player """
        player.reset()

        if clear_scores:
            player.trophies = {
                "bonus" : 0, "enemy0" : 0, "enemy1" : 0, "enemy2" : 0, "enemy3" : 0
            }

        self.shieldPlayer(player, True, 4000)

    def gameOver(self):
        """ End game and return to menu """
        self.game_over = True
        self.running = False
        self.showMenu()

    def showMenu(self):
        """ Show game menu
        Redraw screen only when up or down key is pressed. When enter is pressed,
        exit from this screen and start the game with selected number of players
        """

        global players, screen

        # stop game main loop (if any)
        self.running = False

        # clear all timers
        del gtimer.timers[:]

        # set current stage to 0
        self.stage = 1

        self.nextLevel()

    def reloadPlayers(self):
        """ Init players
        If players already exist, just reset them
        """

        global players

        if len(players) == 0:
            # first player
            x = 8 * self.TILE_SIZE + (self.TILE_SIZE * 2 - 26) / 2
            y = 24 * self.TILE_SIZE + (self.TILE_SIZE * 2 - 26) / 2

            player = Player(
                self.level, 0, [x, y], self.DIR_UP, (0, 0, 13*2, 13*2)
            )
            players.append(player)

            # second player
            if self.nr_of_players == 2:
                x = 16 * self.TILE_SIZE + (self.TILE_SIZE * 2 - 26) / 2
                y = 24 * self.TILE_SIZE + (self.TILE_SIZE * 2 - 26) / 2
                player = Player(
                    self.level, 0, [x, y], self.DIR_UP, (16*2, 0, 13*2, 13*2)
                )
                player.controls = [102, 119, 100, 115, 97]
                players.append(player)

        for player in players:
            player.level = self.level
            self.respawnPlayer(player, True)

    def draw(self):
        global screen, castle, players, enemies, bullets, bonuses

        screen.fill([0, 0, 0])

        self.level.draw([self.level.TILE_EMPTY, self.level.TILE_BRICK, self.level.TILE_STEEL, self.level.TILE_FROZE, self.level.TILE_WATER])

        castle.draw()

        for enemy in enemies:
            enemy.draw()

        for label in labels:
            label.draw()

        for player in players:
            player.draw()

        for bullet in bullets:
            bullet.draw()

        for bonus in bonuses:
            bonus.draw()

        self.level.draw([self.level.TILE_GRASS])
        self.drawSidebar()

        pygame.display.flip()

    def drawSidebar(self):

        global screen, players, enemies

        x = 416
        y = 0
        screen.fill([100, 100, 100], pygame.Rect([416, 0], [64, 416]))

        xpos = x + 16
        ypos = y + 16

        # draw enemy lives
        for n in range(len(self.level.enemies_left) + len(enemies)):
            screen.blit(self.enemy_life_image, [xpos, ypos])
            if n % 2 == 1:
                xpos = x + 16
                ypos+= 17
            else:
                xpos += 17

        # players' lives
        if pygame.font.get_init():
            text_color = pygame.Color('black')
            for n in range(len(players)):
                if n == 0:
                    screen.blit(self.font.render(str(n+1)+"P", False, text_color), [x+16, y+200])
                    screen.blit(self.font.render(str(players[n].lives), False, text_color), [x+31, y+215])
                    screen.blit(self.player_life_image, [x+17, y+215])
                else:
                    screen.blit(self.font.render(str(n+1)+"P", False, text_color), [x+16, y+240])
                    screen.blit(self.font.render(str(players[n].lives), False, text_color), [x+31, y+255])
                    screen.blit(self.player_life_image, [x+17, y+255])

            screen.blit(self.flag_image, [x+17, y+280])
            screen.blit(self.font.render(str(self.stage), False, text_color), [x+17, y+312])

    def toggleEnemyFreeze(self, freeze = True):
        """ Freeze/defreeze all enemies """

        global enemies

        for enemy in enemies:
            enemy.paused = freeze
        self.timefreeze = freeze


    def finishLevel(self):
        """ Finish current level
        Show earned scores and advance to the next stage
        """
        self.active = False
        print("Stage "+str(self.stage)+" completed")
        self.running = False

        self.nextLevel()

    def frame_step(self, input_actions):
        time_passed = self.clock.tick(50)

        pygame.event.pump()
        terminal = False
        reward = 0.1

        if sum(input_actions) != 1:
            raise ValueError('Multiple input actions!')

        for player in players:
            if player.state == player.STATE_ALIVE:
                if input_actions[1] == 1:
                    player.fire()
                elif input_actions[2] == 1:
                    player.move(self.DIR_UP)
                elif input_actions[3] == 1:
                    player.move(self.DIR_RIGHT)
                elif input_actions[4] == 1:
                    player.move(self.DIR_DOWN)
                elif input_actions[5] == 1:
                    player.move(self.DIR_LEFT)
                else:
                    pass

            player.update(time_passed)
        for enemy in enemies:
            if enemy.state == enemy.STATE_DEAD and not self.game_over and self.active:
                enemies.remove(enemy)
                reward += 1
                if len(self.level.enemies_left) == 0 and len(enemies) == 0:
                    self.finishLevel()
                    terminal = True
                    reward += 5
            else:
                enemy.update(time_passed)        
        if not self.game_over and self.active:
            for player in players:
                if player.state == player.STATE_ALIVE:
                    if player.bonus != None and player.side == player.SIDE_PLAYER:
                        self.triggerBonus(bonus, player)
                        player.bonus = None
                        reward += 3
                elif player.state == player.STATE_DEAD:
                    self.superpowers = 0
                    player.lives -= 1
                    if player.lives > 0:
                        self.respawnPlayer(player)
                        reward += -1
                    else:
                        self.gameOver()
                        terminal = True
                        reward += -5

        for bullet in bullets:
            if bullet.state == bullet.STATE_REMOVED:
                bullets.remove(bullet)
            else:
                bullet.update()

        for bonus in bonuses:
            if bonus.active == False: 
                bonuses.remove(bonus)

        for label in labels:
            if not label.active:
                labels.remove(label)

        if not self.game_over:
            if not castle.active:
                self.gameOver()
                terminal = True
                reward += -5
        self.draw()

        image_data = pygame.surfarray.array3d(pygame.display.get_surface())
        image_data = image_data[0:416][0:416]
        image_data = cv2.resize(image_data, (80, 80))

        gtimer.update(time_passed)
        
        return image_data, reward, terminal




