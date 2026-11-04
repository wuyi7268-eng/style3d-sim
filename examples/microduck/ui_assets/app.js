'use strict';
const $ = id => document.getElementById(id);
const token = document.querySelector('meta[name="workbench-token"]').content;
let config, jobs = {},
  runs = [],
  selectedRun = '',
  lastLog = '',
  noticeTimer;
const stateNames = {
  running: '运行中',
  stopping: '正在停止',
  stopped: '已停止',
  completed: '已完成',
  failed: '运行失败'
};
let selectedTraining = '', selectedLog = '', pendingTrain = false;
const active = id => jobs[id] && ['running', 'stopping'].includes(jobs[id].state);
const targetId = kind => kind === 'train' ? selectedTraining :
  kind === 'preview' ? (selectedTraining ? `preview:${selectedTraining}` : '') : kind;
const jobLabel = job => `${job.kind === 'train' ? '训练' : job.kind === 'preview' ? '预览' :
  job.kind === 'evaluation' ? '评估' : 'TensorBoard'} · ${job.name} · ${job.id.slice(-6)}`;
const getStatus = () => api('status?log_id=' + encodeURIComponent(selectedLog));

async function api(path, data) {
  const options = {
    headers: {
      'X-Workbench-Token': token
    }
  };
  if (data !== undefined) {
    options.method = 'POST';
    options.headers['Content-Type'] = 'application/json';
    options.body = JSON.stringify(data);
  }
  const response = await fetch('/api/' + path, options);
  const value = await response.json();
  if (!response.ok) throw new Error(value.error || '操作失败');
  return value;
}

function notify(message, success = false) {
  clearTimeout(noticeTimer);
  $('notice').textContent = message;
  $('notice').className = success ? 'success' : '';
  $('notice').hidden = false;
  if (success) noticeTimer = setTimeout(() => {
    $('notice').hidden = true;
  }, 6500);
}

function formData(form) {
  const values = Object.fromEntries(new FormData(form));
  form.querySelectorAll('input[type=checkbox]').forEach(input => {
    values[input.name] = input.checked;
  });
  return values;
}

function showPage(page) {
  document.querySelectorAll('.page').forEach(element => {
    element.hidden = element.id !== page;
  });
  document.querySelectorAll('[data-page]').forEach(element => {
    element.classList.toggle('active', element.dataset.page === page);
  });
  $('breadcrumb').textContent = page === 'training' ? '训练工作台' : '模型与评估';
  if (page === 'evaluation') refreshModels().catch(error => notify(error.message));
}

function updateDefaults() {
  if (!config) return;
  const profile = config.profiles[$('environment_profile').value];
  const ppo = config.ppo[$('task').value];
  $('num_envs').placeholder = `默认 ${profile.num_envs}（env.py）`;
  $('pad_probability').placeholder = String(profile.pad_probability);
  $('environment-note').textContent = `环境默认 ${profile.num_envs} 个世界 · 物理步长 ${profile.physics.physics_dt} s · ` +
    `非线性 / 线性迭代 ${profile.physics.nonlinear_iterations} / ${profile.physics.linear_iterations}。留空跟随 env.py。`;
  for (const key of ['learning_rate', 'entropy_coef', 'num_learning_epochs', 'num_mini_batches']) {
    $(key).placeholder = String(ppo.algorithm[key]);
  }
  for (const key of ['num_steps_per_env', 'save_interval']) $(key).placeholder = String(ppo[key]);
  $('schedule').options[0].textContent = `任务默认（${ppo.algorithm.schedule}）`;
  const mode = $('mode').value;
  $('source-fields').hidden = mode === 'new';
  $('train-checkpoint').required = mode !== 'new';
  $('restore-label').hidden = mode !== 'resume';
  if (mode !== 'resume') $('restore-label').querySelector('input').checked = false;
  const descriptions = {
    new: '从零开始：随机初始化策略和价值网络，不读取旧权重，优化器与轮数从头开始。适合新任务或基线实验。',
    resume: '严格续训：恢复策略、价值网络、优化器、归一化统计和训练轮数。适合同一实验中断后继续；' +
      '任务、后端、配方及环境必须兼容，PPO 参数沿用权重。下方轮数表示本次追加量，不恢复完整仿真状态。',
    transfer: '迁移初始化：保留已有策略、价值网络、动作标准差及归一化统计，重置优化器和训练轮数。' +
      '适合用平地策略学习软垫，可设置新环境和学习率。目前仅支持关闭自碰撞的 SynReal / velocity；不保证旧能力不退化。'
  };
  $('initialization-help').textContent = descriptions[mode];
  $('ppo-fields').querySelectorAll('input,select').forEach(input => {
    input.disabled = mode === 'resume';
  });
}

function selectTraining(id) {
  selectedTraining = id;
  selectedLog = id;
  renderStatus();
  refreshStatus();
}

function renderExperiments() {
  const trainings = Object.values(jobs).filter(job => job.kind === 'train').reverse();
  if (!jobs[selectedTraining] && trainings.length) selectedTraining = trainings[0].id;
  if (!jobs[selectedLog]) selectedLog = selectedTraining || Object.keys(jobs)[0] || '';
  $('training-count').textContent = `${trainings.filter(job => active(job.id)).length} 个运行中 / ${trainings.length} 个实验`;
  const list = $('training-list');
  const ids = new Set(trainings.map(job => job.id));
  for (const element of [...list.children]) {
    if (!ids.has(element.dataset.id)) element.remove();
  }
  if (!trainings.length && !list.children.length) {
    const empty = document.createElement('p');
    empty.className = 'empty';
    empty.textContent = '尚未启动实验。下方配置每次启动一个独立训练，可同时运行多个。';
    list.append(empty);
  }
  for (const job of trainings) {
    let button = [...list.children].find(element => element.dataset.id === job.id);
    if (!button) {
      button = document.createElement('button');
      button.type = 'button';
      button.dataset.id = job.id;
      button.append(document.createElement('strong'), document.createElement('small'), document.createElement('small'));
      button.addEventListener('click', () => selectTraining(job.id));
      list.prepend(button);
    }
    button.className = 'training-item' + (job.id === selectedTraining ? ' selected' : '');
    button.setAttribute('aria-pressed', job.id === selectedTraining ? 'true' : 'false');
    button.children[0].textContent = `${job.name} · ${stateNames[job.state]}`;
    const cfg = job.settings;
    button.children[1].textContent = `${cfg.solver} / ${cfg.task} · ${cfg.device} · ${cfg.num_envs || '默认'} 环境`;
    const progress = job.metrics.iteration === undefined ? '等待日志' :
      `${job.metrics.iteration + 1} / ${job.metrics.total} 轮`;
    button.children[2].textContent = `${progress} · PID ${job.pid} · ID ${job.id.slice(-6)}` +
      (active(`preview:${job.id}`) ? ' · 预览已打开' : '');
  }
  const logSelect = $('log-kind');
  const signature = Object.keys(jobs).join(',');
  if (logSelect.dataset.signature !== signature) {
    logSelect.replaceChildren();
    for (const job of Object.values(jobs)) logSelect.append(new Option(jobLabel(job), job.id));
    if (!signature) logSelect.append(new Option('暂无进程', ''));
    logSelect.dataset.signature = signature;
  }
  logSelect.value = selectedLog;
}

function renderStatus() {
  renderExperiments();
  const training = jobs[selectedTraining];
  const preview = jobs[targetId('preview')];
  $('selected-training').textContent = training ? `当前查看：${training.name} · ${training.id.slice(-6)}` : '未选择实验';
  $('selected-details').textContent = training?.run_dir || '指标、预览与停止操作对应所选实验；下方表单用于启动新实验。';
  $('train-state').textContent = training ? stateNames[training.state] : '尚未开始';
  $('train-pid').textContent = training ? `PID ${training.pid} · ${Math.floor(training.elapsed / 60)} 分钟` : '等待启动新实验';
  const metrics = training?.metrics || {};
  $('iteration').textContent = metrics.iteration === undefined ? '—' : `${metrics.iteration + 1} / ${metrics.total}`;
  $('fps').textContent = metrics.fps === undefined ? '—' : metrics.fps.toLocaleString();
  $('reward').textContent = metrics.reward === undefined ? '—' : metrics.reward.toFixed(3);
  for (const kind of ['preview', 'evaluation']) {
    const job = jobs[targetId(kind)];
    $(kind + '-state').textContent = job ? stateNames[job.state] : '未打开';
    $(kind + '-state').classList.toggle('running', !!active(targetId(kind)));
  }
  $('start-train').disabled = pendingTrain;
  $('open-preview').disabled = !!active(targetId('preview')) || !training?.run_dir;
  $('start-eval').disabled = !!active('evaluation');
  $('start-tensorboard').disabled = !!active('tensorboard');
  document.querySelectorAll('[data-stop]').forEach(button => {
    const id = targetId(button.dataset.stop);
    button.disabled = !active(id) || jobs[id].state === 'stopping';
  });
  $('preview-checkpoint').textContent = preview ?
    '预览权重：' + preview.command[preview.command.indexOf('--checkpoint') + 1] :
    training?.run_dir ? '当前实验：' + training.run_dir : '请选择实验并等待首次权重保存';
  const log = jobs[selectedLog];
  const text = log?.log ?? (log ? '正在读取所选进程日志…' : '尚无运行日志。配置实验后，启动一个新训练。');
  if (text !== lastLog) {
    $('log-output').textContent = text;
    lastLog = text;
    if ($('auto-scroll').checked) $('log-output').scrollTop = $('log-output').scrollHeight;
  }
  $('log-path').textContent = log?.log_path || '日志与模型保存在本机';
  $('log-exit').textContent = log?.exit_code != null ? `退出码 ${log.exit_code}` : '';
}

function mergeStatus(next) {
  // Ignore out-of-order responses after rapid selection changes; logs never move between IDs.
  for (const [id, job] of Object.entries(next)) {
    if (job.log == null && jobs[id]?.log != null) job.log = jobs[id].log;
    if (job.state === 'failed' && jobs[id]?.state !== 'failed') {
      notify(`${jobLabel(job)} 运行失败，请选择它的日志查看原因。`);
    }
  }
  jobs = next;
  $('connection').textContent = '● 本机已连接';
  $('connection').className = 'badge running';
  renderStatus();
}
let statusSequence = 0;
async function refreshStatus() {
  const sequence = ++statusSequence;
  try {
    const next = await getStatus();
    if (sequence === statusSequence) mergeStatus(next);
  } catch (_) {
    $('connection').textContent = '连接中断 · 正在重试';
    $('connection').className = 'badge';
  }
}
async function poll() {
  await refreshStatus();
  setTimeout(poll, 1500);
}
async function action(button, path, data, kind, message) {
  button.disabled = true;
  if (path === 'train') pendingTrain = true;
  try {
    const result = await api(path, data);
    if (kind && result.id) selectedLog = result.id;
    if (path === 'train') selectedTraining = result.id;
    await refreshStatus();
    notify(message, true);
  } catch (error) {
    notify(error.message);
  } finally {
    if (path === 'train') pendingTrain = false;
    button.disabled = false;
    renderStatus();
  }
}

function renderRuns() {
  const search = $('run-search').value.toLowerCase();
  $('run-list').replaceChildren();
  const filtered = runs.filter(run => `${run.name} ${run.task} ${run.solver}`.toLowerCase().includes(search));
  $('run-count').textContent = `${runs.length} 个实验`;
  if (!filtered.length) {
    const empty = document.createElement('p');
    empty.className = 'empty';
    empty.textContent = runs.length ? '没有符合条件的实验' : '尚未发现模型。完成首次保存后，实验将显示在这里。';
    $('run-list').append(empty);
  }
  for (const run of filtered) {
    const button = document.createElement('button');
    button.className = 'run-button' + (run.path === selectedRun ? ' selected' : '');
    const title = document.createElement('strong');
    title.textContent = run.name;
    const detail = document.createElement('small');
    detail.textContent = `${run.solver} / ${run.task} · ${run.num_envs ?? '—'} 环境 · ${run.checkpoints.length} 个权重`;
    button.append(title, detail);
    button.addEventListener('click', () => selectRun(run));
    $('run-list').append(button);
  }
}

function selectRun(run) {
  selectedRun = run.path;
  $('model-select').replaceChildren();
  for (const path of run.checkpoints) {
    const option = document.createElement('option');
    option.value = path;
    option.textContent = path.split(/[\\/]/).pop();
    $('model-select').append(option);
  }
  if (!run.checkpoints.length) $('model-select').append(new Option('这个实验还没有完整权重', ''));
  $('eval-checkpoint').value = run.checkpoints[0] || '';
  renderRuns();
}
async function refreshModels() {
  runs = await api('models');
  // Refresh the list without silently replacing a manually selected older checkpoint.
  const current = $('eval-checkpoint').value;
  const run = runs.find(item => item.path === selectedRun);
  if (run) {
    selectRun(run);
    if (run.checkpoints.includes(current)) $('model-select').value = current;
    $('eval-checkpoint').value = current;
  } else renderRuns();
}
document.querySelectorAll('[data-page]').forEach(button => {
  button.addEventListener('click', () => showPage(button.dataset.page));
});
for (const id of ['task', 'environment_profile', 'mode']) $(id).addEventListener('change', updateDefaults);
$('environment_profile').addEventListener('change', () => {
  if ($('environment_profile').value !== 'official') {
    $('solver').value = 'synreal';
    updateDefaults();
  }
});
$('train-form').addEventListener('submit', event => {
  event.preventDefault();
  action($('start-train'), 'train', formData(event.target), 'train', '训练进程已启动，初始化进度请看日志。');
});
$('eval-form').addEventListener('submit', event => {
  event.preventDefault();
  action($('start-eval'), 'evaluate', formData(event.target), 'evaluation', '评估正在初始化，3D 窗口将在本机桌面打开。');
});
$('open-preview').addEventListener('click', () => {
  action($('open-preview'), 'preview', {train_id: selectedTraining}, 'preview', '正在打开所选实验的最新权重预览。');
});
document.querySelectorAll('[data-stop]').forEach(button => {
  button.addEventListener('click', () => action(button, 'stop', {
    job_id: targetId(button.dataset.stop)
  }, null, '已请求停止所选进程，其他实验继续运行。'));
});
$('start-tensorboard').addEventListener('click', () => {
  action($('start-tensorboard'), 'tensorboard', {}, 'tensorboard', 'TensorBoard 正在启动，稍后点击「打开」。');
});
$('refresh-models').addEventListener('click', () => refreshModels().catch(error => notify(error.message)));
$('run-search').addEventListener('input', renderRuns);
$('model-select').addEventListener('change', () => {
  $('eval-checkpoint').value = $('model-select').value;
});
$('use-source').addEventListener('click', () => {
  const path = $('eval-checkpoint').value.trim();
  if (!path) return notify('请先选择一个模型权重。');
  $('train-checkpoint').value = path;
  $('mode').value = 'resume';
  const run = runs.find(item => item.checkpoints.includes(path));
  if (run) {
    $('solver').value = run.solver;
    $('task').value = run.task;
    $('environment_profile').value = config.profiles[run.terrain] ? run.terrain : 'official';
  }
  updateDefaults();
  showPage('training');
  notify('已填入来源权重。请核对续训 / 迁移方式以及环境参数后启动。', true);
});
$('log-kind').addEventListener('change', () => {
  selectedLog = $('log-kind').value;
  renderStatus();
  refreshStatus();
});
$('download-log').addEventListener('click', () => {
  const blob = new Blob([jobs[selectedLog]?.log || ''], {
    type: 'text/plain;charset=utf-8'
  });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = `microduck-${$('log-kind').value}-tail.log`;
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
});
async function initialize() {
  try {
    config = await api('config');
    if (config.api_version !== 2) {
      throw new Error('后端仍为旧版，请先在终端重启控制台，再刷新页面。已有训练请先妥善停止');
    }
    for (const task of config.tasks) $('task').append(new Option(task, task));
    $('model-root').textContent = config.log_root;
    $('auth-note').textContent = config.synreal_ready ?
      'SynReal 凭据已在本机进程中配置，实际登录结果见运行日志。' :
      '使用 SynReal：请在终端用 --login 启动控制台并隐藏输入凭据。MuJoCo 无需登录。';
    updateDefaults();
    await refreshModels();
    poll();
  } catch (error) {
    notify('初始化失败：' + error.message + '。请刷新页面重试。');
  }
}
initialize();
