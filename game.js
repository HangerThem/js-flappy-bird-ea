const canvas = document.getElementById("gameCanvas");
const ctx = canvas.getContext("2d");

const canvasNeural = document.getElementById("neuralNetworkCanvas");
const ctxNeural = canvasNeural.getContext("2d");

function randomColor() {
  return `rgb(${Math.random() * 255}, ${Math.random() * 255}, ${
    Math.random() * 255
  })`;
}

class Bird {
  constructor(color) {
    this.x = canvas.width / 3;
    this.y = canvas.height / 2;
    this.vy = 0;
    this.width = 20;
    this.height = 20;
    this.weight = 1;
    this.isDead = false;
    this.color = color;
    this.score = 0;
    this.timeAlive = 0;
    this.fitness = 0;
    this.brain = new NeuralNetwork(5, 8, 1);
  }

  think(pipes) {
    let closestPipe = pipes.find((pipe) => pipe.x + pipe.width > this.x);
    let inputs = [
      this.y / canvas.height,
      closestPipe.top / canvas.height,
      closestPipe.bottom / canvas.height,
      closestPipe.x / canvas.width,
      this.vy / 10,
    ];
    let output = this.brain.predict(inputs);
    if (output[0] > 0.5) {
      this.flap();
    }
  }

  update() {
    if (this.y > canvas.height - this.height || this.y < 0) {
      this.y = canvas.height - this.height;
      this.vy = 0;
      this.isDead = true;
    } else {
      this.timeAlive++;
      this.vy += this.weight;
      this.y += this.vy;
    }
  }

  draw() {
    if (this.isDead) return;
    ctx.fillStyle = this.color;
    ctx.fillRect(this.x, this.y, this.width, this.height);
    ctx.fillStyle = "white";
    ctx.fillText(this.score, this.x, this.y);
  }

  flap() {
    this.vy -= 10;
  }

  die() {
    this.isDead = true;
  }

  copy() {
    let bird = new Bird(this.color);
    bird.brain = this.brain.copy();
    return bird;
  }
}

class Pipe {
  constructor(x, y) {
    this.x = x;
    this.y = y;
    this.width = 20;
    this.top = Math.floor(Math.random() * 150 + 150);
    this.bottom = canvas.height - this.top - 200;
    this.color = "green";
  }

  update() {
    this.x -= 10;
  }

  draw() {
    ctx.fillStyle = this.color;
    ctx.fillRect(this.x, this.y, this.width, this.top);
    ctx.fillRect(this.x, canvas.height - this.bottom, this.width, this.bottom);
  }

  isColliding(bird) {
    if (
      bird.x < this.x + this.width &&
      bird.x + bird.width > this.x &&
      (bird.y < this.y + this.top ||
        bird.y + bird.height > canvas.height - this.bottom)
    ) {
      bird.die();
    }
  }

  isOffScreen() {
    return this.x + this.width < 0;
  }

  static generatePipes() {
    const pipes = [];
    for (let i = 0; i < 3; i++) {
      pipes.push(new Pipe(canvas.width + i * 300, 0));
    }
    return pipes;
  }

  static updatePipes(pipes, birds) {
    pipes.forEach((pipe) => pipe.update());
    Pipe.drawPipes(pipes);
    Pipe.checkCollision(birds, pipes);
    Pipe.removeOffScreenPipes(pipes);
    Pipe.addNewPipe(pipes);
    Pipe.updateScore(birds, pipes);
    return pipes;
  }

  static drawPipes(pipes) {
    pipes.forEach((pipe) => pipe.draw());
  }

  static checkCollision(birds, pipes) {
    pipes.forEach((pipe) => {
      birds.forEach((bird) => {
        pipe.isColliding(bird);
      });
    });
  }

  static removeOffScreenPipes(pipes) {
    return pipes.filter((pipe) => !pipe.isOffScreen());
  }

  static addNewPipe(pipes) {
    if (pipes[pipes.length - 1].x < canvas.width - 300) {
      pipes.push(new Pipe(canvas.width, 0));
    }
  }

  static updateScore(birds, pipes) {
    pipes.forEach((pipe) => {
      birds.forEach((bird) => {
        if (bird.x > pipe.x && bird.x < pipe.x + pipe.width) {
          bird.score++;
        }
      });
    });
  }
}

class NeuralNetwork {
  constructor(inputNodes, hiddenNodes, outputNodes) {
    this.inputNodes = inputNodes;
    this.hiddenNodes = hiddenNodes;
    this.outputNodes = outputNodes;
    this.weights_ih = new Matrix(this.hiddenNodes, this.inputNodes);
    this.weights_ho = new Matrix(this.outputNodes, this.hiddenNodes);
    this.weights_ih.randomize();
    this.weights_ho.randomize();
    this.bias_h = new Matrix(this.hiddenNodes, 1);
    this.bias_o = new Matrix(this.outputNodes, 1);
    this.bias_h.randomize();
    this.bias_o.randomize();
    this.learningRate = 0.1;
    this.hiddenActivations = null;
    this.outputActivations = null;
  }

  predict(inputArray) {
    let inputs = Matrix.fromArray(inputArray);
    let hidden = Matrix.multiply(this.weights_ih, inputs);
    hidden.add(this.bias_h);
    hidden.map(sigmoid);
    this.hiddenActivations = hidden;
    let output = Matrix.multiply(this.weights_ho, hidden);
    output.add(this.bias_o);
    output.map(sigmoid);
    this.outputActivations = output;
    return output.toArray();
  }

  copy() {
    let nn = new NeuralNetwork(
      this.inputNodes,
      this.hiddenNodes,
      this.outputNodes
    );
    nn.weights_ih = this.weights_ih.copy();
    nn.weights_ho = this.weights_ho.copy();
    nn.bias_h = this.bias_h.copy();
    nn.bias_o = this.bias_o.copy();
    nn.learningRate = this.learningRate;
    return nn;
  }

  mutate(rate) {
    function mutate(val) {
      if (Math.random() < rate) {
        return val + Math.random() * 0.1 - 0.05;
      } else {
        return val;
      }
    }
    this.weights_ih.map(mutate);
    this.weights_ho.map(mutate);
    this.bias_h.map(mutate);
    this.bias_o.map(mutate);
  }

  draw() {
    ctxNeural.clearRect(0, 0, canvasNeural.width, canvasNeural.height);
    ctxNeural.font = "10px Arial";
    ctxNeural.fillStyle = "black";
    const spacing = 50;
    const radius = 15;
    const inputX = 50;
    const hiddenX = 150;
    const outputX = 250;
    const y = 50;
    this.drawLayer(inputX, y, this.inputNodes, radius, spacing);
    this.drawLayer(
      hiddenX,
      y,
      this.hiddenNodes,
      radius,
      spacing,
      this.hiddenActivations
    );
    this.drawLayer(
      outputX,
      y,
      this.outputNodes,
      radius,
      spacing,
      this.outputActivations
    );
    this.drawConnections(
      inputX,
      hiddenX,
      this.weights_ih,
      this.inputNodes,
      this.hiddenNodes,
      radius,
      spacing
    );
    this.drawConnections(
      hiddenX,
      outputX,
      this.weights_ho,
      this.hiddenNodes,
      this.outputNodes,
      radius,
      spacing
    );
  }

  drawLayer(x, y, nodes, radius, spacing, activations = null) {
    for (let i = 0; i < nodes; i++) {
      ctxNeural.beginPath();
      ctxNeural.arc(x, y + i * spacing, radius, 0, Math.PI * 2);
      if (activations) {
        const activation = activations.data[i][0];
        ctxNeural.fillStyle = `rgba(${activation * 255 + 50}, 0, 0, 1)`;
      } else {
        ctxNeural.fillStyle = "white";
      }
      ctxNeural.fill();
      ctxNeural.fillStyle = "white";
      activations?.data[i][0] &&
        ctxNeural.fillText(
          activations?.data[i][0].toFixed(2),
          x - radius / 2 - 2,
          y + i * spacing
        );
    }
  }

  drawConnections(x1, x2, weights, nodes1, nodes2, radius, spacing) {
    for (let i = 0; i < nodes1; i++) {
      for (let j = 0; j < nodes2; j++) {
        let val = weights.data[j][i];
        ctxNeural.strokeStyle = val > 0 ? "blue" : "red";
        ctxNeural.lineWidth = Math.abs(val) * 5;
        ctxNeural.beginPath();
        ctxNeural.moveTo(x1 + radius, 50 + i * spacing);
        ctxNeural.lineTo(x2 - radius, 50 + j * spacing);
        ctxNeural.stroke();
      }
    }
  }
}

function sigmoid(x) {
  return 1 / (1 + Math.exp(-x));
}

class Matrix {
  constructor(rows, cols) {
    this.rows = rows;
    this.cols = cols;
    this.data = Array(this.rows)
      .fill()
      .map(() => Array(this.cols).fill(0));
  }

  randomize() {
    this.data = this.data.map((row) => row.map(() => Math.random() * 1.4 - 1));
  }

  static fromArray(arr) {
    let m = new Matrix(arr.length, 1);
    m.data = arr.map((el) => [el]);
    return m;
  }

  toArray() {
    return this.data.flat();
  }

  add(n) {
    if (n instanceof Matrix) {
      if (this.rows !== n.rows || this.cols !== n.cols) {
        console.error(
          "Columns and Rows of A must match Columns and Rows of B."
        );
        return;
      }
      this.data = this.data.map((row, i) =>
        row.map((col, j) => col + n.data[i][j])
      );
    } else {
      this.data = this.data.map((row) => row.map((col) => col + n));
    }
  }

  static subtract(a, b) {
    let result = new Matrix(a.rows, a.cols);
    result.data = result.data.map((row, i) =>
      row.map((col, j) => a.data[i][j] - b.data[i][j])
    );
    return result;
  }

  static multiply(a, b) {
    if (a.cols !== b.rows) {
      console.error("Columns of A must match rows of B.");
      return;
    }
    let result = new Matrix(a.rows, b.cols);
    result.data = result.data.map((row, i) =>
      row.map((col, j) =>
        a.data[i].reduce((acc, el, k) => acc + el * b.data[k][j], 0)
      )
    );
    return result;
  }

  map(fn) {
    this.data = this.data.map((row) => row.map((col) => fn(col)));
  }

  copy() {
    let m = new Matrix(this.rows, this.cols);
    m.data = this.data.map((row) => row.slice());
    return m;
  }
}

const POPULATION_SIZE = 2000;
const MUTATION_RATE = 0.2;

class GeneticAlgorithm {
  constructor() {
    this.birds = Array(POPULATION_SIZE)
      .fill()
      .map(() => new Bird(randomColor()));
    this.pipes = Pipe.generatePipes();
    this.generation = 1;
    this.bestBird = this.birds[0];
    this.timeout = 1000;
  }

  update() {
    this.pipes = Pipe.updatePipes(this.pipes, this.birds);
    this.birds.forEach((bird) => {
      bird.think(this.pipes);
      bird.update();
    });
    this.timeout--;
    this.checkAllDead();
    this.checkTimeout();
  }

  draw() {
    this.birds.forEach((bird) => bird.draw());
    Pipe.drawPipes(this.pipes);
    this.drawScore();
  }

  checkAllDead() {
    if (this.birds.every((bird) => bird.isDead)) {
      this.timeout = Math.max(this.generation * 500, 3000);
      this.nextGeneration();
    }
  }

  checkTimeout() {
    if (this.timeout <= 0) {
      this.timeout = Math.max(this.generation * 500, 3000);
      this.nextGeneration();
    }
  }

  nextGeneration() {
    this.calculateFitness();
    this.birds = this.generateNewPopulation();
    this.pipes = Pipe.generatePipes();
    this.generation++;
  }

  calculateFitness() {
    let totalScore = this.birds.reduce(
      (acc, bird) => acc + bird.score + bird.timeAlive,
      0
    );
    this.birds.forEach(
      (bird) => (bird.fitness = (bird.score + bird.timeAlive) / totalScore)
    );
    this.birds.sort((a, b) => b.fitness - a.fitness);
    if (!this.bestBird || this.birds[0].score > this.bestBird.score) {
      this.bestBird = this.birds[0];
    }
  }

  generateNewPopulation() {
    let newPopulation = [];
    for (let i = 0; i < POPULATION_SIZE - 1; i++) {
      let parentA = this.selectParentTournament();
      let parentB = this.selectParentRandom();
      let child = this.crossover(parentA, parentB);
      child.color = randomColor();
      child.score = 0;
      child.timeAlive = 0;
      child.brain.mutate(MUTATION_RATE);
      newPopulation.push(child);
    }
    newPopulation.push(this.bestBird.copy());
    return newPopulation;
  }

  crossover(parentA, parentB) {
    let child = parentA.copy();
    let split = Math.floor(Math.random() * child.brain.weights_ih.data.length);
    for (let i = split; i < child.brain.weights_ih.data.length; i++) {
      child.brain.weights_ih.data[i] = parentB.brain.weights_ih.data[i];
    }
    return child;
  }

  selectParentTournament() {
    let tournamentSize = 5;
    let tournament = [];
    for (let i = 0; i < tournamentSize; i++) {
      tournament.push(
        this.birds[Math.floor(Math.random() * this.birds.length)]
      );
    }
    tournament.sort((a, b) => b.fitness - a.fitness);
    return tournament[0];
  }

  selectParentRoulette() {
    let r = Math.random();
    let index = 0;
    while (r > 0) {
      r -= this.birds[index].fitness;
      index++;
    }
    index--;
    return this.birds[index];
  }

  selectParentRandom() {
    return this.birds[Math.floor(Math.random() * this.birds.length)];
  }

  selectParentStochasticUniversalSampling() {
    let r = Math.random();
    let index = 0;
    let sum = this.birds.reduce((acc, bird) => acc + bird.fitness, 0);
    while (r > 0) {
      r -= this.birds[index].fitness;
      index++;
    }
    index--;
    return this.birds[index];
  }

  selectParentRank() {
    let r = Math.random();
    let index = 0;
    let sum = this.birds.reduce((acc, bird, i) => acc + i, 0);
    while (r > 0) {
      r -= this.birds[index].fitness;
      index++;
    }
    index--;
    return this.birds[index];
  }

  drawScore() {
    ctx.fillStyle = "white";
    ctx.font = "20px Arial";
    ctx.fillText(`Generation: ${this.generation}`, 10, 30);
    ctx.fillText(`Timeout: ${this.timeout}`, 10, 60);
    ctx.fillText(
      `Population: ${this.birds.filter((bird) => !bird.isDead).length}`,
      10,
      90
    );
    if (this.bestBird) {
      ctx.fillText(`Best Fitness: ${this.bestBird.fitness}`, 10, 120);
      ctx.fillText(`Best Score: ${this.bestBird.score}`, 10, 150);
    }
  }
}

const ga = new GeneticAlgorithm();

function gameLoop() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ga.update();
  ga.draw();
  ga.birds[0].brain.draw();
  requestAnimationFrame(gameLoop);
}

gameLoop();
